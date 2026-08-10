# The labelling app: technical specification

Implements `PLAN.md` §5. One Next.js app on Vercel, one Postgres database, no other moving
parts. Written to be built from without further design decisions.

**The problem this design exists to solve, stated once.** `LABELING_GUIDE.md` opens with the
number that governs everything here: the same construct names score **κ = 0.13–0.30 from cold
crowdworkers and κ = 0.65–0.71 from in-house annotators given anchors, worked examples and a
calibration pilot.** Handing a public link to lots of people is, by default, a machine for
producing the first regime. Every feature below — the calibration gate, the always-visible
anchors, one metric at a time, the required span, the injected checks — exists to buy back that
3–5×, because a probe cannot beat the noise in its own labels and 20,000 labels at κ = 0.2 are
worth less than 2,000 at κ = 0.65. Anything that raises throughput at the cost of agreement is a
net loss and should be rejected on those grounds.

Everything marked **`GUESS`** is an assumption about intent; §9 indexes them all.

---

## 0. Shape of the system

```
tutor-labeller/                      one Vercel project, root = projects/tutor_metrics/app
  app/
    page.tsx                         landing + consent + start
    label/page.tsx                   the labelling screen (client component)
    calibration/page.tsx             the 15-item quiz
    me/page.tsx                      rater's own progress
    admin/page.tsx                   dashboard (bearer-token gated)
    api/…/route.ts                   every route in §5, Node runtime
  lib/
    db.ts                            Pool + withTx
    session.ts                       cookie issue / lookup
    metrics.ts                       loads the metric table, caches per instance
    assign.ts                        the claim query
    export.ts                        §7 transform
    qc.ts                            §4 statistics
    types.ts                         §5.1
  migrations/001_init.sql … 00N_*.sql
  scripts/migrate.ts  seed_metrics.ts  ingest_turns.ts
  instrumentation-client.ts          initBotId
  next.config.ts                     withBotId
  vercel.ts                          crons
```

No queue, no Redis, no worker, no ORM, no separate API service. Cron jobs are route handlers in
the same app. Roughly fifteen SQL queries in total.

---

## 1. Data model

Postgres 16 on Neon. One migration file, applied by `scripts/migrate.ts`.

### 1.1 Two rules the schema enforces structurally

**Provenance never reaches the server.** `build_blind_set.py` already splits a pool from a key and
keeps the key on disk; this design does the same. The ingest script computes the salted turn id
locally, uploads only blind-safe fields, and keeps `key.json` on the researcher's laptop. The
database therefore cannot leak the model, style, temperature, sample index or probe score,
because it never receives them. What it does receive is an opaque `stratum` string so batches can
be balanced without the balancing telling anyone anything.

**Nulls are never zeros.** `LABELING_GUIDE.md` §9 is emphatic: a construct the rater called
inapplicable has no presence to average, and filling it with 0 converts "the question did not
apply" into "the artifact did it badly". `label.value` is nullable, `na_reason` is non-null
exactly when it is, and the export omits the key rather than emitting a null.

### 1.2 DDL

```sql
-- migrations/001_init.sql
create extension if not exists pgcrypto;

-- ─────────────────────────────────────────────────────────────────── content

create type turn_pool as enum ('production', 'gold');

create table dialogue (
  id          text primary key,            -- salted locally; opaque here
  subject     text,
  grade       int,
  question    text        not null,
  choices     jsonb,                       -- ["$2.09", …] or null for free response
  gold        text,                        -- the correct option / answer
  reference   text,                        -- worked solution; see build_pilot.py
  created_at  timestamptz not null default now()
);

create table turn (
  id             text        primary key,  -- THE EXPORT ID. Salted, stable, opaque.
  dialogue_id    text        not null references dialogue(id) on delete cascade,
  pool           turn_pool   not null default 'production',
  turn_index     int         not null,
  context        jsonb       not null default '[]'::jsonb,  -- [{role,text}] before this turn
  student_before text        not null,
  tutor_turn     text        not null,
  n_words        int         not null,     -- computed at ingest
  stratum        text,                     -- opaque bucket for balanced sampling
  created_at     timestamptz not null default now()
);
create index turn_dialogue_idx on turn (dialogue_id);
create index turn_pool_idx     on turn (pool);

-- The ONLY relation rater-facing routes are allowed to read turn text from.
create view turn_public as
  select t.id, t.turn_index, t.context, t.student_before, t.tutor_turn, t.n_words,
         d.question, d.choices, d.gold, d.reference, d.subject, d.grade
    from turn t join dialogue d on d.id = t.dialogue_id;

-- ─────────────────────────────────────────────────────────────────── rubric

create type metric_scale as enum ('ordinal', 'nonordinal', 'categorical');
create type metric_group as enum ('A', 'B', 'C', 'guard', 'covariate');

create table metric (
  key             text         primary key,
  version         int          not null default 1,
  grp             metric_group not null,   -- `group` is reserved; TS calls this field `group`
  question        text         not null,   -- exactly the rater-facing wording
  anchors         jsonb        not null,   -- [{"value":1,"label":"…"}, …] ascending
  guidance        text,                    -- the `?` overlay: boundary cases, worked examples
  scale           metric_scale not null default 'ordinal',
  lo              smallint     not null default 1,
  hi              smallint     not null default 3,
  requires_span   boolean      not null default true,
  span_from_value smallint,                -- span demanded only at value >= this; null = always
  needs_reference boolean      not null default false,
  needs_dialogue  boolean      not null default true,
  allow_na        boolean      not null default false,
  source_sha      text         not null,   -- sha256 of the metrics.py definition
  active          boolean      not null default true,
  created_at      timestamptz  not null default now(),
  check (hi > lo)
);

-- Wording changes retroactively reinterpret existing labels, so every one is kept.
create table metric_revision (
  key        text        not null,
  version    int         not null,
  question   text        not null,
  anchors    jsonb       not null,
  guidance   text,
  source_sha text        not null,
  note       text,
  changed_at timestamptz not null default now(),
  primary key (key, version)
);

create or replace function metric_snapshot() returns trigger language plpgsql as $$
begin
  if tg_op = 'UPDATE'
     and (new.question is distinct from old.question
       or new.anchors  is distinct from old.anchors
       or new.guidance is distinct from old.guidance) then
    if new.version <= old.version then
      raise exception 'metric %: wording changed without bumping version', new.key;
    end if;
  end if;
  insert into metric_revision (key, version, question, anchors, guidance, source_sha)
  values (new.key, new.version, new.question, new.anchors, new.guidance, new.source_sha)
  on conflict do nothing;
  return new;
end $$;
create trigger metric_snapshot_t after insert or update on metric
  for each row execute function metric_snapshot();

-- ─────────────────────────────────────────────────────────────────── raters

create type rater_status as enum
  ('new', 'calibrating', 'active', 'probation', 'suspended', 'failed_calibration');

create table rater (
  id                   uuid         primary key default gen_random_uuid(),
  handle               text         unique not null,  -- 'r_7fb3c2a1'; the export filename
  status               rater_status not null default 'new',
  cohort               text,                          -- ?src= on the landing link
  is_researcher        boolean      not null default false,
  email                text,                          -- only if volunteered; nullable forever
  created_at           timestamptz  not null default now(),
  last_seen_at         timestamptz  not null default now(),
  calibration_attempts int          not null default 0,
  calibration_score    numeric,
  qualified_at         timestamptz,
  n_labels             int          not null default 0,
  n_gold_seen          int          not null default 0,
  n_gold_correct       int          not null default 0,
  qc_computed_at       timestamptz,
  qc_note              text
);

create table rater_metric (           -- per-rater, per-metric bookkeeping
  rater_id   uuid not null references rater(id) on delete cascade,
  metric_key text not null references metric(key),
  primed_at  timestamptz,             -- non-null once they've done the 2-item primer
  n_labels   int  not null default 0,
  primary key (rater_id, metric_key)
);

create table rater_session (
  id           uuid        primary key default gen_random_uuid(),
  rater_id     uuid        not null references rater(id) on delete cascade,
  token_hash   bytea       not null unique,   -- sha256(cookie token); the token is never stored
  created_at   timestamptz not null default now(),
  last_seen_at timestamptz not null default now(),
  expires_at   timestamptz not null default now() + interval '30 days',
  revoked_at   timestamptz,
  ip_hash      bytea,                          -- sha256(ip || IP_HASH_PEPPER). No raw IPs.
  country      text,
  user_agent   text,
  n_labels     int         not null default 0
);
create index rater_session_rater_idx on rater_session (rater_id);

create table rater_status_event (
  id          bigserial primary key,
  rater_id    uuid        not null references rater(id) on delete cascade,
  from_status rater_status,
  to_status   rater_status not null,
  reason      text         not null,
  at          timestamptz  not null default now()
);

-- ─────────────────────────────────────────────────────────── work: items

create table batch (
  id             text        primary key,
  note           text,
  overlap_target numeric     not null default 0.20,
  active         boolean     not null default false,
  created_at     timestamptz not null default now()
);

-- One row per (turn, metric). This is the unit of work, because the rubric is
-- answered one metric at a time and kappa is computed per metric.
create table item (
  id            bigserial primary key,
  batch_id      text      not null references batch(id) on delete cascade,
  turn_id       text      not null references turn(id) on delete cascade,
  metric_key    text      not null references metric(key),
  target_labels smallint  not null default 1 check (target_labels between 1 and 4),
  n_labels      smallint  not null default 0,
  claimed_until timestamptz,
  claimed_by    uuid      references rater(id),
  priority      int       not null default 0,
  shuffle       double precision not null default random(),
  unique (batch_id, turn_id, metric_key)
);

create index item_serve_idx
  on item (batch_id, metric_key, ((n_labels > 0)) desc, priority, shuffle)
  where n_labels < target_labels;
create index item_progress_idx on item (batch_id, metric_key, n_labels, target_labels);
create index item_turn_idx     on item (turn_id);

-- ─────────────────────────────────────────────────────── work: known answers

create table gold_item (
  id            bigserial primary key,
  turn_id       text      not null references turn(id) on delete cascade,
  metric_key    text      not null references metric(key),
  gold_value    smallint  not null,
  tolerance     smallint  not null default 0,   -- credit when |answer - gold| <= tolerance
  rationale     text      not null,             -- one sentence, shown after the answer
  role          text      not null check (role in ('positive_canonical','positive_atypical',
                    'positive_partial','hard_negative','says_only','control')),
  use_in_quiz   boolean   not null default false,
  quiz_position int,
  source        text      not null default 'authored'
                  check (source in ('authored','earned')),
  active        boolean   not null default true,
  n_served      int       not null default 0,
  n_correct     int       not null default 0,
  created_at    timestamptz not null default now(),
  unique (turn_id, metric_key)
);

-- An AUTHORED answer and the analysed pool must never overlap: a pool that labels its
-- own answers measures the labeller's belief about the answer. Earned gold (§4.1) is
-- derived from production labels, so its turn legitimately stays in the production pool.
create or replace function assert_pool() returns trigger language plpgsql as $$
declare p turn_pool;
begin
  select pool into p from turn where id = new.turn_id;
  if tg_table_name = 'item' and p <> 'production' then
    raise exception 'turn % is pool=%; cannot be a production item', new.turn_id, p;
  end if;
  if tg_table_name = 'gold_item' and new.source = 'authored' and p <> 'gold' then
    raise exception 'turn % is pool=%; cannot hold an authored gold answer', new.turn_id, p;
  end if;
  return new;
end $$;
create trigger item_pool_t      before insert on item
  for each row execute function assert_pool();
create trigger gold_item_pool_t before insert on gold_item
  for each row execute function assert_pool();

-- ────────────────────────────────────────────────────── work: assignments

create type assignment_kind as enum ('production', 'quiz', 'primer');

create table assignment (
  id           uuid            primary key default gen_random_uuid(),
  rater_id     uuid            not null references rater(id) on delete cascade,
  session_id   uuid            not null references rater_session(id) on delete cascade,
  batch_id     text            references batch(id),
  metric_key   text            references metric(key),   -- null only for the mixed quiz
  kind         assignment_kind not null default 'production',
  n_items      int             not null,
  n_done       int             not null default 0,
  created_at   timestamptz     not null default now(),
  expires_at   timestamptz     not null default now() + interval '45 minutes',
  completed_at timestamptz
);
create index assignment_rater_idx on assignment (rater_id, created_at desc);

create table assignment_item (
  assignment_id   uuid   not null references assignment(id) on delete cascade,
  position        int    not null,
  presentation_id uuid   not null unique default gen_random_uuid(),  -- all the client ever sees
  item_id         bigint references item(id) on delete cascade,
  gold_item_id    bigint references gold_item(id) on delete cascade,
  retest_of       bigint,                                   -- label.id being re-presented
  served_at       timestamptz,
  answered_at     timestamptz,
  primary key (assignment_id, position),
  check (num_nonnulls(item_id, gold_item_id) = 1)
);

-- ─────────────────────────────────────────────────────────────────── labels

create table label (
  id              bigserial   primary key,
  rater_id        uuid        not null references rater(id) on delete cascade,
  session_id      uuid        not null references rater_session(id),
  assignment_id   uuid        not null references assignment(id) on delete cascade,
  presentation_id uuid        not null unique references assignment_item(presentation_id),

  turn_id         text        not null references turn(id),
  metric_key      text        not null references metric(key),
  metric_version  int         not null,

  item_id         bigint      references item(id) on delete cascade,
  gold_item_id    bigint      references gold_item(id) on delete cascade,
  retest_of       bigint      references label(id),

  value           smallint,                     -- null iff not applicable
  na_reason       text check (na_reason in
                    ('missing_reference','missing_context','not_a_tutor_turn','unintelligible')),
  span_start      int,
  span_end        int,
  span_text       text,
  sentence_index  int,
  flag_kind       text check (flag_kind in
                    ('rubric_misfit','says_only','factual_error','broken_item')),
  flag_note       text,

  shown_at        timestamptz not null,         -- server's issue time, not the client's
  answered_at     timestamptz not null default now(),
  dwell_ms        int         not null,         -- server-computed from shown_at
  client_dwell_ms int,                          -- diagnostic only; never used in QC
  revisions       smallint    not null default 0,

  metric_source_sha text      not null,
  guide_version     text      not null,
  ui_version        text      not null,
  created_at        timestamptz not null default now(),

  check ((value is null) = (na_reason is not null)),
  check (num_nonnulls(item_id, gold_item_id) = 1)
);

-- THE constraint that makes double-labelling honest: one rater, one item, once.
create unique index label_one_per_rater_idx
  on label (item_id, rater_id)
  where item_id is not null and retest_of is null;

create index label_rater_item_idx  on label (rater_id, item_id);
create index label_item_idx        on label (item_id) where item_id is not null;
create index label_turn_metric_idx on label (turn_id, metric_key);
create index label_rater_time_idx  on label (rater_id, answered_at desc);
create index label_metric_idx      on label (metric_key, answered_at desc);

-- ───────────────────────────────────────────────────── denormalised counters

create or replace function label_counters() returns trigger language plpgsql as $$
declare
  d   int := -1;
  hit int := 0;
  r   record;
begin
  if tg_op = 'INSERT' then d := 1; r := new; else r := old; end if;

  if r.item_id is not null and r.retest_of is null then
    update item set n_labels = n_labels + d where id = r.item_id;
  end if;

  if r.gold_item_id is not null then
    select case when r.value is not null and abs(r.value - g.gold_value) <= g.tolerance
                then 1 else 0 end
      into hit from gold_item g where g.id = r.gold_item_id;
    update gold_item set n_served  = n_served  + d,
                         n_correct = n_correct + d * hit
                   where id = r.gold_item_id;
    update rater     set n_gold_seen    = n_gold_seen    + d,
                         n_gold_correct = n_gold_correct + d * hit
                   where id = r.rater_id;
  end if;

  update assignment    set n_done   = n_done   + d where id = r.assignment_id;
  update rater_session set n_labels = n_labels + d where id = r.session_id;
  update rater         set n_labels = n_labels + d, last_seen_at = now()
                     where id = r.rater_id;
  update rater_metric  set n_labels = n_labels + d
                     where rater_id = r.rater_id and metric_key = r.metric_key;
  update assignment_item set answered_at = case when d = 1 then now() else null end
                     where presentation_id = r.presentation_id;
  return null;
end $$;
create trigger label_counters_t after insert or delete on label
  for each row execute function label_counters();

-- ───────────────────────────────────────────────────────────────────── QC

create table rater_qc (
  rater_id          uuid        not null references rater(id) on delete cascade,
  computed_at       timestamptz not null default now(),
  metric_key        text        not null default '*',   -- '*' = pooled across metrics
  n_labels          int         not null,
  n_overlap         int         not null default 0,
  kappa_w           numeric,
  gold_acc          numeric,
  median_dwell_ms   int,
  frac_below_floor  numeric,
  mode_share_excess numeric,
  runs_z            numeric,
  length_rho        numeric,
  primary key (rater_id, computed_at, metric_key)
);

create table metric_qc (
  metric_key       text        not null references metric(key),
  computed_at      timestamptz not null default now(),
  n_labels         int         not null,
  n_pairs          int         not null,
  kappa_w          numeric,
  flag_rate        numeric,
  na_rate          numeric,
  length_rho       numeric,
  value_histogram  jsonb       not null,
  primary key (metric_key, computed_at)
);

-- ──────────────────────────────────────────────────────────── admin views

create view metric_progress as
  select i.batch_id, i.metric_key,
         count(*)                                                   as n_items,
         sum(i.target_labels)                                       as target,
         sum(i.n_labels)                                            as got,
         count(*) filter (where i.n_labels >= i.target_labels)      as complete,
         count(*) filter (where i.target_labels = 2)                as pairs_planned,
         count(*) filter (where i.target_labels = 2 and i.n_labels >= 2) as pairs_done
    from item i group by 1, 2;
```

`rater_metric` rows are upserted by `/api/assignment` before the first label on a metric lands,
because `label_counters()` updates that row rather than creating it.

### 1.3 What each field is for

| Requirement from the brief | Where it lives |
|---|---|
| Blinding | `turn.id` salted at ingest; provenance never uploaded; `turn_public` is the only readable relation; `presentation_id` is a fresh uuid per serving |
| Double-labelling overlap | `item.target_labels`, set to 2 on exactly `batch.overlap_target` of rows at materialisation |
| No rater labels an item twice | `label_one_per_rater_idx`, plus a `NOT EXISTS` in the claim query |
| Per-label timing | `label.shown_at` (server), `dwell_ms` (server-computed), `client_dwell_ms` (diagnostic), `revisions` |
| Rater quality | `rater.*` counters, `rater_qc` snapshots, `rater_status_event` audit |
| Calibration | `gold_item.use_in_quiz` + `quiz_position`; `assignment.kind` |
| Instrument quality | `metric_qc.flag_rate`, `na_rate`, `length_rho` — the rubric is on trial too |

---

## 2. Rater flow

### 2.1 Onboarding — three screens, under two minutes

1. **Landing.** What the task is, who it is for, how long a set takes, that nothing personal is
   collected, and one button. Also the consent notice (§6.6). The link carries `?src=` which is
   stored as `rater.cohort`, so recruitment channels can be compared later.
2. **How to answer** — a single screen, not a document. Four rules, taken verbatim from
   `LABELING_GUIDE.md` §1 because they are the ones that move κ:
   - You are judging whether the turn **does** the thing, not whether it is good teaching.
   - Answer by **pointing at the text**. If you have to form an overall impression, flag it.
   - Check factual claims **against the reference solution shown**; do not re-derive them.
     (This one question went from κ 0.25 to 0.75 on that change alone.)
   - You are not judging whether a different response would have been better. Three tutors pick
     the same action in 18% of cases, so that question has no recoverable answer.
3. **Start calibration.**

There is no signup, no email, no password. §6 explains why that is the right trade and how abuse
is handled instead.

### 2.2 Calibration — 15 items, and it teaches rather than filters

Composition, following `LABELING_GUIDE.md` §2.1's item roles. Every item has a known answer and a
one-sentence rationale, and **feedback is shown immediately after each answer**. This is the
calibration session from the guide's §10.4 rendered as software: the 3–5× effect comes from
resolving disagreements into stated rules, not from excluding people.

| Role | n | What it tests |
|---|---|---|
| `positive_canonical` | 4 | Can they recognise the construct at all |
| `positive_atypical` | 2 | Unusual surface form, no trigger vocabulary |
| `positive_partial` | 2 | Does the ordinal carry information |
| `hard_negative` | 4 | Do they discriminate the near-miss |
| `says_only` | 3 | Do they read **topic** where they should read **structure** |

Items span the active metrics; each item is presented exactly like a production item (§3), one
metric at a time, so calibration also teaches the interface.

**Pass mark.** Score = exact matches + 0.5 × within-one on `ordinal` metrics. Pass at **≥ 10.5 of
15 (70%)**, with one absolute override: **two or more of the three `says_only` items marked
present is a fail regardless of total.** That mirrors gate G4 in the guide, which is the strictest
gate there for the same reason — a rater who reads pedagogical vocabulary as pedagogical structure
will produce a confident, plausible, wrong column.

**On failure.** One retake, offered after an anchors review screen, using a *different* 15 items
(hold back a second set of 15). Two failures → `failed_calibration`, told plainly, thanked, shown
the anchors page. No shadow pool, no silently discarded work.

**GUESS**: 15 items with a 70% pass mark, and the `says_only` override. `PLAN.md` specifies "~15
calibration turns with known consensus answers" and nothing about scoring.

### 2.3 The metric primer — 2 items, not a gate

The first time a rater is served metric *M*, the assignment is `kind='primer'`: two known-answer
items on *M* with immediate feedback, then straight into production on the same metric. Recorded
in `rater_metric.primed_at`. They do not count as labels and are never exported.

This is an addition to what `PLAN.md` asks for (**`GUESS`**), and the argument for it is the one
the plan already makes: protocol beat construct selection every time it was tested, and the
cheapest protocol available is showing someone two worked answers to the exact question they are
about to answer two hundred times. It costs about a minute per rater per metric.

### 2.4 The labelling loop

An **assignment** is *one metric × 12 turns*, plus a ~50% chance of one injected check item at a
random position. One metric per assignment is not a preference: `label_ui.py` found that rating
one dimension across many turns is a different and much faster task than rating six on a few, and
`consensus.py` found that asking for several judgements in one context couples them — which is how
`actionable` and `elicits` ended up correlated at 0.96.

```
POST /api/assignment  →  claim 12 items  →  label them  →  POST /api/assignment again
                              ↑                                        │
                              └────────  same metric by default  ◄─────┘
```

After every third assignment (~36 items) a break screen appears, encoding the guide's 40-item /
90-minute session cap. "Keep going" is available but is not the focused button.

### 2.5 How a rater gets turns — the claim query

One statement. Atomic, concurrency-safe, and it enforces the overlap target and the
no-double-labelling rule at the same time.

```sql
-- $1 batch_id, $2 metric_key, $3 rater_id, $4 how many
with picked as (
  select i.id
    from item i
   where i.batch_id   = $1
     and i.metric_key = $2
     and i.n_labels   < i.target_labels
     and (i.claimed_until is null or i.claimed_until < now())
     and not exists (
           select 1 from label l
            where l.item_id = i.id and l.rater_id = $3)
   order by (i.n_labels > 0) desc,   -- finish started pairs first
            i.priority,
            i.shuffle
   limit $4
     for update skip locked          -- must follow LIMIT
)
update item
   set claimed_until = now() + interval '45 minutes',
       claimed_by    = $3
 where id in (select id from picked)
returning id, turn_id;
```

`SKIP LOCKED` filters *after* the limit, so this can return fewer than `$4` rows under
contention. Serve a short assignment rather than retrying; it self-corrects on the next one.

Three things worth reading closely.

**`(i.n_labels > 0) desc` is the overlap engine.** Items already carrying one label sort ahead of
fresh ones, so a `target_labels = 2` row gets its partner at the first opportunity. Completed
*pairs* — the only thing κ can be computed from — accumulate as fast as the rater population
allows, which is what `PLAN.md` §5 means by "so κ can be computed continuously rather than
discovered at the end". The alternative, random order, still reaches 20% eventually but leaves
agreement unmeasurable for most of the round. The previous round shipped slices with **zero**
overlap and agreement was simply unmeasurable; this is the structural fix.

**`for update skip locked` is why two raters never get the same item.** The claim expires after 45
minutes, so an abandoned assignment self-heals with no reaper process. `claimed_until` is a
soft-lock for scheduling, not a correctness mechanism — correctness is the unique index.

**The overlap fraction is data, not code.** It is set once, at materialisation:

```sql
-- POST /api/admin/batch, per active metric
insert into item (batch_id, turn_id, metric_key, target_labels, priority)
select $1, t.id, m.key,
       case when random() < b.overlap_target then 2 else 1 end,
       0
  from turn t
  cross join metric m
  cross join (select overlap_target from batch where id = $1) b
 where t.pool = 'production'
   and t.id = any($2::text[])
   and m.active
   and (not m.needs_reference or exists (
         select 1 from dialogue d where d.id = t.dialogue_id and d.reference is not null))
on conflict do nothing;
```

Raising the overlap later is an `UPDATE`, not a redeploy. A metric whose κ looks shaky can be
given `target_labels = 3` on a slice without touching the app.

### 2.6 Which metric a rater gets

Sticky within a session. On a new assignment: keep the previous metric if it still has work
available to this rater; otherwise pick the active metric with the lowest completion ratio that
has at least 12 servable items for them. The rater can also switch deliberately from `/me`.

```sql
select m.key
  from metric m
  join metric_progress p on p.metric_key = m.key and p.batch_id = $1
 where m.active
   and exists (
         select 1 from item i
          where i.batch_id = $1 and i.metric_key = m.key
            and i.n_labels < i.target_labels
            and (i.claimed_until is null or i.claimed_until < now())
            and not exists (select 1 from label l
                             where l.item_id = i.id and l.rater_id = $2)
          limit 1)
 order by (p.got::numeric / nullif(p.target, 0)) asc, random()
 limit 1;
```

**One tension to be explicit about.** `LABELING_GUIDE.md` §5 says never to label all items of one
construct consecutively, because a run of retrieval-practice items trains you to see retrieval
practice. `PLAN.md` §5 requires one metric at a time. Both are right about different risks. The
guide is warning about a run of *designed positives*; the plan is avoiding cross-metric
contamination. The resolution: keep one metric per assignment, and make the **answer sequence**
unpredictable by stratifying each assignment across the LLM-judge prevalence scores from
`PLAN.md` phase 4, so a rater is never served twelve turns that are all obviously a 1. Set
`item.priority` at materialisation from the judge's predicted value and interleave. If the
prevalence pass has not been run, fall back to pure shuffle and accept the response-set risk.

### 2.7 Preventing repeat labelling

| Case | Handling |
|---|---|
| Same rater, same (turn, metric) | Forbidden. `label_one_per_rater_idx` rejects it at the database; the claim query never offers it. |
| Same rater, same turn, **different metric** | Allowed — otherwise assignment becomes badly constrained at 20 metrics. Recorded implicitly: `(rater_id, turn_id, answered_at)` is enough to derive "had this rater seen this turn before", so an anchoring effect can be tested for at analysis time rather than assumed away. |
| Same rater, same (turn, metric), **deliberately** | The retest block. `retest_of` set, partial unique index does not apply, fresh `presentation_id` so it cannot be matched by lookup. |
| Two sessions, one person, two cookies | Not preventable and not worth trying. `ip_hash` + `user_agent` correlation surfaces it on the admin dashboard as a hint, nothing more. |

### 2.8 Retest

`LABELING_GUIDE.md` §4 calls intra-rater retest "the ceiling on everything else", and the previous
round re-presented nothing at all, so one human's signal could not be separated from her own
noise. Implementation is cheap:

When building an assignment for a rater with ≥ 40 labels on this metric, and with probability
0.05, replace one item with a retest — a label of theirs on this metric, `answered_at < now() -
interval '72 hours'`, chosen at random, re-presented under a fresh `presentation_id` at a
different position. The rater is never told.

**GUESS**: `PLAN.md` does not mention retest for the app. It is included because the guide treats
it as non-optional and it costs one line in the assignment builder. Be honest in the round report
that it only yields data from raters who come back after three days, which on a public link will
be a minority.

---

## 3. The labelling screen

One metric. One turn. Everything needed to answer, nothing else.

### 3.1 Layout

Desktop ≥ 1100 px, two columns. Below that, the anchors move to a sticky bottom sheet and the
reading column goes full width.

```
┌───────────────────────────────────────────────────────────────────────────┐
│  locates_student_object      ▓▓▓▓▓▓▓░░░░░  7/12      break in 29    ?     │  sticky, 56px
├──────────────────────────────────────────────┬────────────────────────────┤
│  READING COLUMN            max 72ch          │  QUESTION COLUMN    420px  │  sticky
│                                              │                            │
│  ▸ The problem              collapsed        │  Does it point at a        │
│  ▸ Worked solution          iff needs_ref    │  specific number, step     │
│  ▸ Earlier in the dialogue  collapsed        │  or word the student       │
│  ──────────────────────────────────────      │  wrote?                    │
│  STUDENT, JUST BEFORE        15px, dim       │                            │
│  I added 7 + 98 + 204 and got 309.           │  ┌──────────────────────┐  │
│                                              │  │1│ Points at nothing… │  │
│  ▍ TUTOR TURN — rate this    18px, ink       │  ├──────────────────────┤  │
│  ¹Good, the sum is right. ²Now look at       │  │2│ Points at the pro… │  │
│  the units — you wrote 309 with no           │  ├──────────────────────┤  │
│  dollar sign.                                │  │3│ Points at a speci… │  │
│                                              │  └──────────────────────┘  │
│                                              │   n  not applicable        │
│                                              │   f  flag  ·  u  undo      │
└──────────────────────────────────────────────┴────────────────────────────┘
```

Which panels appear is driven by the metric row, not hard-coded: `needs_reference` shows the
worked solution expanded, `needs_dialogue` shows earlier turns collapsed. The student's previous
message is **always** shown and never collapsible — several metrics are defined against it.

### 3.2 Readability, treated as a first-order requirement

Raters do hundreds of these. Fatigue is the mechanism by which agreement decays.

**Type.** System UI stack. Base 16 px. Tutor turn **18 px / 1.65**, which is the only text read
word-by-word. Student message 15 px, problem and reference 14.5 px. Reading column capped at
**72ch**; the tutor turn itself sits at ~62ch. `white-space: pre-wrap` so the model's own line
breaks survive, matching `label_ui.py`.

**Contrast.** Both themes ship; the choice is remembered per rater in `localStorage`. Body text
clears WCAG AAA (7:1), secondary clears AA (4.5:1).

| | Background | Body text | Secondary |
|---|---|---|---|
| Dark (default) | `#10131a` | `#e9ecf3` — **15.6:1** | `#a0a9ba` — **7.8:1** |
| Light | `#fcfcfd` | `#16181d` — **17.2:1** | `#565e70` — **6.3:1** |

Accent `#6ea8fe` (dark) / `#1f5fd0` (light) is used only for the turn's left rule and the focused
anchor. Selected-anchor state is carried by background **and** a check glyph, never by colour
alone.

**No layout shift, ever.** This is the single biggest ergonomic win at volume, and it costs
discipline rather than cleverness:

- The whole 12-item assignment is fetched in one response. Advancing to the next item is a state
  change, not a network round-trip. No spinners between items.
- The tutor-turn panel has a `min-height` set to the 90th-percentile turn height, so short turns
  do not pull the anchor rows upward.
- Anchor rows are fixed-height with the label clamped to three lines and the full text in the `?`
  overlay. The three rows are at the same *y* on every item, so muscle memory works.
- The only transition is a 90 ms opacity fade on the turn text. Nothing slides.

**Fatigue.** Progress is shown per assignment (12, attainable), not per corpus (thousands,
demoralising). The corpus total lives on `/me`. A break screen at every 36 items.

### 3.3 The span requirement

`LABELING_GUIDE.md` §3.2 calls the required span "the single rule most responsible for the
difference between the two reliability regimes". Typing a quotation is far too much friction at
three hundred items an hour, so the span is selected with **one keystroke**.

The tutor turn is pre-split into sentences server-side and rendered with superscript indices.
After a rater presses a score that requires evidence, the anchor panel is replaced *in place* —
same position, no scroll — by:

```
Which part does it?      ¹  ²  ³        (or drag to select)
```

They press `1`, `2` or `3`. Two keystrokes total. Character offsets, sentence index and text are
all stored. Dragging a native selection overrides the sentence choice for cases where the
sentence is the wrong unit. Turns with a single sentence auto-select and skip the step.

`metric.span_from_value` controls when it is demanded: for a 1–3 ordinal, `span_from_value = 2`
asks for evidence whenever the rater claims the thing is present at all, and never for a 1. Guards
are `requires_span = false` — there is no span to point at for the absence of something, which is
the same asymmetry that put them outside the summed reward in the first place.

### 3.4 Keyboard map

Copied from `label_ui.py` where it already works, extended where it must.

| Key | Action |
|---|---|
| `1`–`6` | Score, within the metric's `lo`–`hi`. Out-of-range keys are ignored, not clamped. |
| `1`–`9` (span step) | Choose the sentence. |
| `n` | Not applicable, then `1`–`4` for the reason. Only when `metric.allow_na`. |
| `f` | Flag. Opens a 4-way picker plus an optional note. `Esc` closes. |
| `u` | Undo the last answer and step back one item. |
| `←` `→` | Back / forward within the assignment. Forward only over answered items. |
| `?` | Overlay: full anchors, `metric.guidance`, the four rules from §2.1. |
| `Enter` | **Nothing.** Deliberately inert, so a stray return cannot submit. |

While a text input has focus, number keys are ignored and `Esc` blurs — `label_ui.py` already gets
this right and it is easy to regress.

### 3.5 The client contract

- The assignment arrives whole; the client holds it in memory and POSTs each answer as it is
  given. An answer is durable the moment it is accepted, exactly as `label_ui.py` writes the file
  after every keystroke. Closing the tab loses at most the answer in flight.
- A failed POST retries three times with backoff, then shows a persistent banner and blocks
  further answers. Silent loss of labels is the worst possible failure here.
- `shown_at` for QC is the **server's** `assignment_item.served_at`, stamped when the assignment
  is built and refined per item by the order of arrival. The client's own timing is stored as
  `client_dwell_ms` and used for nothing that matters, because it is trivially forged.

---

## 4. Quality control

Two subjects are on trial: the rater, and the rubric. The second matters more — a metric with a
20% flag rate is broken however diligent everyone is.

### 4.1 Ongoing checks

Injected known-answer items, drawn from the gold pool, indistinguishable from production items,
at roughly **1 in 24** (a 50% chance of one per 12-item assignment). No feedback is shown; the
rater does not learn which ones they were.

The gold pool has to be bigger than the quiz or regulars will recognise it, so it grows itself:
nightly, any (turn, metric) whose `target_labels` are complete and where **three or more qualified
raters agreed exactly** is promoted to `gold_item` with `source = 'earned'`.

The promoted turn stays in the production pool and its three real labels stay in the export —
only the *check* labels served against it are excluded, by virtue of carrying `gold_item_id`
instead of `item_id`. That is the point of allowing `source = 'earned'` past the `assert_pool`
trigger: an earned check is not a hidden answer key, it is a fourth opinion on a settled item, and
nothing circular reaches the analysis.

One caveat for the round report: earned gold measures agreement with the majority, not
correctness, so a rater who is right where three others were wrong is scored down for it. Require
unanimity rather than a majority, cap at ~200 per metric, and treat `gold_acc` as a screen that
puts a rater in front of a human rather than as a verdict.

### 4.2 Detecting a rater who is clicking through

Computed nightly per rater, and per (rater, metric), into `rater_qc`.

| Statistic | Definition | Why this one |
|---|---|---|
| `frac_below_floor` | Fraction of labels with `dwell_ms < 900 + 45 × n_words` | 45 ms/word is ~1,300 wpm — three times faster than the fastest real reader. This is a *cannot have read it* detector, not a diligence meter, and it should almost never fire on an honest rater. |
| `mode_share_excess` | Their modal-value share on metric *m*, minus the pool's | Raw mode share is useless: `length_fit` was legitimately 85% one value. The excess over the pool is the signal. |
| `runs_z` | Wald–Wolfowitz z on the mode / not-mode sequence | Catches "1,1,1,1,1,1,1,2,1,1,1,1" which mode share alone can miss. Flag at z < −3. |
| `gold_acc` | `n_gold_correct / n_gold_seen` | Direct. |
| `kappa_w` | Linearly weighted κ against the rounded mean of the other raters, on their double-labelled items | The number that actually matters. Linear weights, matching `agreement.py`. |
| `length_rho` | Spearman(value, log word count), per metric | Straight from `LABELING_GUIDE.md` S9. A rater whose every metric tracks length is scoring length. Flag when \|ρ\| > 0.6 *and* the pool's ρ for that metric is < 0.3. |
| `median_dwell_ms` | — | Context for everything above. |

None of these is decisive alone. `gold_acc` and `kappa_w` are the two that carry weight; the rest
are screens that raise a rater onto the dashboard for a human to look at.

### 4.3 Status machine

| Transition | Condition | Effect |
|---|---|---|
| `active → probation` | any of: `gold_acc < 0.60` (n ≥ 8); `kappa_w < 0.20` (n ≥ 20); `frac_below_floor > 0.20` (n ≥ 30); `mode_share_excess > 0.30` (n ≥ 40); `runs_z < −3` (n ≥ 40) | Labels still collected, marked. Rater sees nothing. |
| `probation → active` | last 50 labels clear all of the above | Marked recovered. |
| `probation → suspended` | 50 further labels still failing, or `gold_acc < 0.35` at n ≥ 8 | No new assignments. Shown a plain message. |
| any → `suspended` | manual, from the dashboard | Reason recorded in `rater_status_event`. |

**Labels are never deleted, and status is never applied retroactively to the data.** The export
filters by status at export time (`?min_status=active`), so the effect of excluding a cohort can
be *measured* rather than assumed. This is the same discipline as `LABELING_GUIDE.md` §8.4 — do
not adjudicate away a low κ, because the low κ is the finding. Deleting a suspended rater's labels
destroys the only evidence about how much they mattered.

### 4.4 Do you show raters their own agreement?

**Known-answer feedback: yes.** In calibration and the primer, immediately and in full — that is
the mechanism, not a courtesy. In production, `/me` shows a coarse gold-check accuracy band
(`most checks correct` / `some checks missed`), updated only every 20 items so a single check
cannot be back-inferred.

**Peer agreement: no, not while collection is open.** Showing a rater their κ against others turns
the task from *read the anchors and answer* into *guess what the others said*. Agreement then
measures conformity to a displayed target, is inflated by an unknown amount, and cannot be
corrected for afterwards — and inter-rater agreement is the exact quantity the whole project needs
to be able to trust. It is published to everyone once the round closes.

### 4.5 Watching the rubric, not just the raters

`metric_qc`, nightly, on the admin dashboard, with the guide's standing rules wired in as alerts:

- **`flag_rate > 0.20` on a metric → rewrite or drop it, whatever its κ.** A high flag rate is a
  low κ arriving more politely.
- `na_rate` high → the required context is missing from the corpus, not from the rater.
- `length_rho` for the *pool* per metric, alongside `kappa_w`. This is the number `PLAN.md` is
  built around; it should be visible from day one, not discovered at analysis.
- Value histogram per metric. A metric coming back 90% one value has a prevalence problem that
  phase 4 was supposed to catch, and more labelling will not fix it.

---

## 5. API routes

All routes are Next.js App Router handlers on the **Node runtime** (Fluid Compute default). No
Edge runtime anywhere.

Auth kinds: **public** (no credential), **rater** (valid session cookie), **qualified** (rater
with `status in ('active','probation')`), **admin** (`Authorization: Bearer $ADMIN_TOKEN`),
**cron** (`Authorization: Bearer $CRON_SECRET`).

### 5.1 Core types

```ts
// lib/types.ts

export type MetricScale = 'ordinal' | 'nonordinal' | 'categorical';
export type MetricGroup = 'A' | 'B' | 'C' | 'guard' | 'covariate';
export type RaterStatus =
  | 'new' | 'calibrating' | 'active' | 'probation' | 'suspended' | 'failed_calibration';
export type NaReason =
  'missing_reference' | 'missing_context' | 'not_a_tutor_turn' | 'unintelligible';
export type FlagKind = 'rubric_misfit' | 'says_only' | 'factual_error' | 'broken_item';

export interface Anchor { value: number; label: string }

export interface Metric {
  key: string;
  version: number;
  group: MetricGroup;
  question: string;
  anchors: Anchor[];              // ascending by value, length = hi - lo + 1
  guidance: string | null;
  scale: MetricScale;
  lo: number;
  hi: number;
  requiresSpan: boolean;
  spanFromValue: number | null;
  needsReference: boolean;
  needsDialogue: boolean;
  allowNa: boolean;
}

export interface Sentence { index: number; start: number; end: number; text: string }

export interface TurnPublic {
  id: string;                     // the salted id; also the export id
  question: string;
  choices: string[] | null;
  gold: string | null;            // shown: `correct` and `leak`-shaped metrics need it
  reference: string | null;       // worked solution; check against, do not re-derive
  context: Array<{ role: 'student' | 'tutor'; text: string }>;
  studentBefore: string;
  tutorTurn: string;
  sentences: Sentence[];          // for one-keystroke span selection
}

export interface AssignmentItemPayload {
  presentationId: string;         // the ONLY id the client ever sees
  position: number;
  turn: TurnPublic;
}

export interface Assignment {
  id: string;
  kind: 'production' | 'quiz' | 'primer';
  metric: Metric;                 // one metric for the whole assignment
  items: AssignmentItemPayload[];
  expiresAt: string;              // ISO
  progress: { doneInSet: number; doneTotal: number; sinceBreak: number };
}

export interface LabelSubmission {
  presentationId: string;
  value: number | null;                        // null iff naReason is set
  naReason?: NaReason;
  span?: { start: number; end: number; text: string; sentenceIndex: number | null };
  flag?: { kind: FlagKind; note?: string };
  clientDwellMs: number;
  revisions: number;
}

export interface CalibrationFeedback {
  correct: boolean;
  goldValue: number;
  rationale: string;
  running: { correct: number; answered: number; of: number };
}

export interface LabelAccepted {
  ok: true;
  progress: { doneInSet: number; doneTotal: number; sinceBreak: number };
  breakSuggested: boolean;
  feedback?: CalibrationFeedback;              // only when kind is 'quiz' | 'primer'
}

export interface RaterState {
  handle: string;
  status: RaterStatus;
  qualified: boolean;
  nLabels: number;
  goldBand: 'good' | 'mixed' | 'unknown';      // coarse and delayed, see §4.4
  metrics: Array<{ key: string; nLabels: number; primed: boolean }>;
  calibration: { attempts: number; score: number | null; passMark: number } | null;
}
```

### 5.2 Routes

| # | Route | Method | Auth | Input | Output |
|---|---|---|---|---|---|
| 1 | `/api/session` | POST | public + BotID | `{ src?: string }` | `{ rater: RaterState }`, sets cookie |
| 2 | `/api/me` | GET | rater | — | `RaterState` |
| 3 | `/api/calibration/next` | GET | rater | — | `{ item: AssignmentItemPayload, metric: Metric, index, of } \| { done, score, passed }` |
| 4 | `/api/calibration/answer` | POST | rater | `LabelSubmission` | `LabelAccepted` (feedback always present) |
| 5 | `/api/assignment` | POST | qualified + BotID | `{ metricKey?: string, size?: number }` | `Assignment` |
| 6 | `/api/assignment/current` | GET | rater | — | `Assignment \| null` |
| 7 | `/api/label` | POST | qualified | `LabelSubmission` | `LabelAccepted` |
| 8 | `/api/label/undo` | POST | qualified | `{ presentationId: string }` | `{ ok: true }` |
| 9 | `/api/metric/[key]` | GET | rater | — | `Metric` |
| 10 | `/api/health` | GET | public | — | `{ ok, db, metricShaOk, version }` |
| 11 | `/api/admin/stats` | GET | admin | `?batch=` | `AdminStats` |
| 12 | `/api/admin/raters` | GET | admin | `?status=&sort=` | `RaterQcRow[]` |
| 13 | `/api/admin/rater/[id]` | PATCH | admin | `{ status, reason }` | `{ ok: true }` |
| 14 | `/api/admin/turns` | POST | admin | `{ dialogues: […], turns: […] }` | `{ inserted, skipped }` |
| 15 | `/api/admin/batch` | POST | admin | `{ batchId, turnIds, overlapTarget, metrics? }` | `{ items }` |
| 16 | `/api/admin/gold` | POST | admin | `{ goldItems: […] }` | `{ inserted }` |
| 17 | `/api/admin/export` | GET | admin | `?batch=&min_status=&format=` | `ExportBundle` or `{ url }` |
| 18 | `/api/cron/qc` | GET | cron | — | `{ ratersScored, metricsScored, transitions }` |
| 19 | `/api/cron/snapshot` | GET | cron | — | `{ blobUrl, bytes }` |

### 5.3 The two routes with real logic

**`POST /api/assignment`** — the only route needing a multi-statement transaction.

```ts
export async function POST(req: NextRequest) {
  if ((await checkBotId()).isBot)
    return NextResponse.json({ error: 'denied' }, { status: 403 });

  const { rater, session } = await requireQualified(req);
  const { metricKey, size = 12 } = await req.json();

  return withTx(async (tx) => {
    await tx.query(
      `update assignment set completed_at = now()
        where rater_id = $1 and completed_at is null`, [rater.id]);

    const metric = metricKey ?? await pickMetric(tx, BATCH, rater.id);
    if (!metric) return NextResponse.json({ error: 'no_work' }, { status: 409 });

    const primed = await isPrimed(tx, rater.id, metric);
    const kind: AssignmentKind = primed ? 'production' : 'primer';

    const claimed = kind === 'primer'
      ? await pickPrimerGold(tx, metric, rater.id, 2)
      : await claimItems(tx, BATCH, metric, rater.id, size);      // §2.5, one statement
    if (!claimed.length) return NextResponse.json({ error: 'no_work' }, { status: 409 });

    const rows = kind === 'production'
      ? injectCheck(claimed, await maybeGold(tx, metric, rater.id))   // ~50% of the time
      : claimed;

    const a  = await tx.one(`insert into assignment
        (rater_id, session_id, batch_id, metric_key, kind, n_items)
        values ($1,$2,$3,$4,$5,$6) returning id, expires_at`,
        [rater.id, session.id, BATCH, metric, kind, rows.length]);

    const ai = await tx.many(`insert into assignment_item
        (assignment_id, position, item_id, gold_item_id, retest_of, served_at)
        select $1, ord, i.item_id, i.gold_item_id, i.retest_of, now()
          from jsonb_to_recordset($2::jsonb)
               as i(ord int, item_id bigint, gold_item_id bigint, retest_of bigint)
        returning position, presentation_id, item_id, gold_item_id`,
        [a.id, JSON.stringify(rows)]);

    return NextResponse.json(await hydrate(tx, a, metric, ai));   // reads turn_public only
  });
}
```

**`POST /api/label`** — one insert, everything else is the trigger.

Validation, in order, all server-side:

1. `presentation_id` exists, belongs to this rater's current assignment, and `answered_at is null`.
2. Assignment not expired.
3. `value` is null XOR `na_reason` is null; if non-null, `lo ≤ value ≤ hi` for the metric.
4. `na_reason` only if `metric.allow_na`.
5. Span present when `metric.requires_span` and `value >= metric.span_from_value`; span offsets
   inside the turn; `span_text` matches the substring at those offsets. A client-supplied span
   that does not match the stored turn text is rejected — that is the whole point of the rule.
6. `dwell_ms = now() - assignment_item.served_at`, server-side. The client's number is stored
   separately and never used for QC.
7. Insert. A unique-violation on `label_one_per_rater_idx` returns `409 duplicate` rather than 500.

**Nothing is rejected for being too fast.** A 200 ms answer is stored with its 200 ms and handled
by §4. Rejecting it would teach an abuser the threshold and create a second code path for no
analytic gain.

---

## 6. Auth and abuse

The link is public. The design goal is: zero friction for a person who wants to help, and no
economic reason for anyone else to bother.

### 6.1 Rater identity — an anonymous signed session, not an account

`POST /api/session` mints a rater row with a random handle (`r_` + 4 random bytes hex) and a
session token: 32 bytes from `crypto.randomBytes`, returned in a cookie, stored only as
`sha256(token)`.

```
Set-Cookie: tl_s=<base64url token>; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=2592000
```

Lookup is one indexed query on `token_hash`. There is no password, no reset flow, no third-party
redirect, no PII, and nothing to leak. This is a *session*, not an auth system — the reason to
prefer hosted auth is that credential handling is easy to get wrong, and there are no credentials
here. Signup of any kind, even a magic link, would cost more labels than it saves garbage: a
public labelling link converts on the order of a few percent, and an email wall roughly halves
that. (**`GUESS`** — no conversion data exists for this project.)

Optional, off by default: a rater can attach an email on `/me` to reclaim their handle on another
device. Nothing depends on it.

### 6.2 Admin

`Authorization: Bearer $ADMIN_TOKEN`, compared with `crypto.timingSafeEqual`. One researcher, one
token, five lines. This is a deliberate departure from "prefer a hosted option" (**`GUESS`** that
it is the right one): provisioning Clerk to gate a dashboard with exactly one user is more moving
parts, not fewer. Layer Vercel Deployment Protection over the project if belt-and-braces is
wanted. If named accounts are ever needed, Clerk via the Vercel Marketplace is the drop-in.

### 6.3 Bots

Vercel BotID, cost-aware. Deep Analysis is billed per `checkBotId()` call, so it goes where the
call count is small and the value is high:

| Route | Check | Volume |
|---|---|---|
| `POST /api/session` | `checkBotId()` | ~1 per rater |
| `POST /api/assignment` | `checkBotId()` | ~1 per 12 labels |
| `POST /api/label` | none — cookie + WAF only | the hot path |

At 100,000 labels that is roughly 8,000 Deep Analysis calls, about $8 on Pro. Gating every label
would be $100 for no extra protection, since a client that cleared the assignment check already
holds a valid session.

```ts
// next.config.ts
import { withBotId } from 'botid/next/config';
export default withBotId({ /* … */ });
```

```ts
// instrumentation-client.ts
import { initBotId } from 'botid/client/core';
initBotId({ protect: [
  { path: '/api/session',    method: 'POST' },
  { path: '/api/assignment', method: 'POST' },
]});
```

Note the local-development behaviour: `checkBotId()` returns `isBot: false` locally unless
`developmentOptions` is set, and testing a protected route with `curl` will be blocked in
production. Test through a `fetch` from the app.

### 6.4 Rate limits

Two layers, doing different jobs.

**Vercel WAF — a crude ceiling against floods.** Stage every rule with `--action log` first and
review the dashboard before enforcing. Counters are per region, so the effective limit can be a
few times the configured one.

```bash
vercel firewall rules add "Session flood" \
  --condition '{"type":"path","op":"eq","value":"/api/session"}' \
  --condition '{"type":"method","op":"eq","value":"POST"}' \
  --action rate_limit --rate-limit-window 3600 --rate-limit-requests 20 \
  --rate-limit-keys ip --rate-limit-action deny --yes

vercel firewall rules add "Label flood" \
  --condition '{"type":"path","op":"eq","value":"/api/label"}' \
  --action rate_limit --rate-limit-window 60 --rate-limit-requests 600 \
  --rate-limit-keys ip --rate-limit-action log --yes

vercel firewall rules add "Admin surface" \
  --condition '{"type":"path","op":"pre","value":"/api/admin"}' \
  --action rate_limit --rate-limit-window 60 --rate-limit-requests 60 \
  --rate-limit-keys ip --rate-limit-action deny --yes
```

**The label limit is deliberately loose, and that is not an oversight.** A seminar room or a
school on one NAT is exactly the audience this app is for; an IP-keyed limit tight enough to catch
one abusive rater would take out the whole room. 600/min per IP catches a script and nothing else.

**The app — the limit that means something.** Per rater, in SQL, checked in `/api/assignment`:
one open assignment at a time; a hard cap of 1,500 labels per rolling 24 hours; a break screen
every 36. Per rater is the right key because the cookie, not the IP, is what identifies a rater.

### 6.5 What is actually protected

| Asset | Protection |
|---|---|
| Gold answers | `gold_value` is never serialised outside calibration/primer feedback, and only after the label row exists. |
| Item enumeration | The client sees only `presentation_id`: a random uuid, single-use, scoped to one assignment and one rater. No sequential id ever leaves the server. |
| Provenance | Not in the database (§1.1). Nothing to exfiltrate. |
| Turn text | Not a secret. Someone scraping tutor turns has cost you nothing. |
| The bill | Set a Vercel spend limit. Blocked WAF traffic is not billed. |

### 6.6 Consent and data, which is not optional to think about

The app collects: answers, timings, a country code, a truncated user-agent, and a salted IP hash.
No name, no email unless volunteered, no free text about the rater. The landing page states this
in four lines and links to a one-paragraph notice.

**`GUESS`, and worth checking before launch:** the assumption here is that this is unpaid
volunteer labelling of model-generated text with no human-subjects component, so a short notice
suffices. If raters are recruited as students, compensated through a platform, or if the data
will appear in a paper that an IRB reviews, a proper consent screen and an ethics determination
are required, and retro-fitting consent onto labels already collected is not possible.

---

## 7. Export

The requirement is that `extract_hidden.py` and `fit_head.py` run **unchanged**. I read both, plus
`agreement.load` which `fit_head` imports, to fix the shape exactly.

### 7.1 What the consumers actually read

| Consumer | Reads | Required fields |
|---|---|---|
| `extract_hidden.py --units` | `blob["units"]` (or a bare list) | `id`, `question`, `student_before`, `tutor_turn` |
| `fit_head.py --slices` (`word_counts`) | `blob["units"]` | `id`, `tutor_turn` |
| `fit_head.py --labels` → `agreement.load` | one file per rater | `blob["rater"]`, `blob["labels"] = [{ "id", "<metric>": int, … }]` |
| `agreement.py --labels` | same | same, plus optional `blob["shots"]` |

Three properties of that pipeline the export depends on, all verified in the source:

- `fit_head` collects `[r[dim.key] for r in by_unit[uid].values() if isinstance(r.get(dim.key), int)]`
  and takes `fmean`. **A missing key is silently skipped.** One-metric-at-a-time labelling produces
  sparse per-rater records, and the existing pipeline handles them natively — no change needed.
- `agreement.load` builds `{unit_id: {rater: record}}`. **Two records from the same rater for the
  same turn overwrite each other.** Retests must therefore go in a separate file or they destroy
  the primary label.
- `rubric.validate` reads `record["flag"]` as a **string**, whereas `LABELING_GUIDE.md` uses a list.
  The export emits a string.

### 7.2 The bundle

`GET /api/admin/export?batch=b1&min_status=active` returns one JSON envelope. A 30-line
`scripts/unpack_export.py` writes it to disk — chosen over a zip so there is no archive dependency
in the function and no streaming to get wrong.

```
data/
  label_slices/units.json    {"schema","shared","units":[Unit]}       ← extract_hidden, --slices
  labels/<handle>.json       {"schema","rater","shots","labels"}      ← fit_head, agreement
  covariates/<handle>.json   same shape; covariate + non-ordinal keys ← NEVER globbed into labels/
  retest/<handle>.json       same shape; second-pass labels only
  spans/<handle>.jsonl       {turn_id, metric_key, value, span_text, sentence_index}
  flags.jsonl                {turn_id, metric_key, rater, flag_kind, flag_note}
  manifest.json              counts, versions, filters, metric shas, n_synthetic
```

`spans/` is not decoration: it is the audit trail for the says-only failure mode. A metric whose
positive spans are mostly the sentence that *names* the construct rather than the sentence that
*performs* it is being read as topic, and that is visible in this file and nowhere else.

`flags.jsonl` is the unlossy record; the `flag` string inside `labels/` is the lossy projection the
existing loader expects.

```json
// data/label_slices/units.json
{
  "schema": "pedagogy-rm/v1",
  "shared": 412,
  "units": [
    {
      "id": "b7c1f0a93e4d21",
      "question": "While cleaning his room, Paul found 7 cents…",
      "choices": ["$2.09", "$3.09", "$3.72", "$4.08"],
      "gold": "$3.09",
      "reference": "7 + 98 = 105 cents…",
      "student_before": "I added 7 + 98 + 204 and got 309.",
      "tutor_turn": "Good, the sum is right. Now look at the units…",
      "turn_index": 5,
      "subject": "math",
      "grade": 5
    }
  ]
}
```

```json
// data/labels/r_7fb3c2a1.json
{
  "schema": "pedagogy-rm/labels-v1",
  "rater": "r_7fb3c2a1",
  "shots": {},
  "labels": [
    { "id": "b7c1f0a93e4d21", "locates_student_object": 3, "diagnoses_the_error": 2 },
    { "id": "b0a2d51c77ff90", "locates_student_object": 1,
      "flag": "locates_student_object: rubric_misfit — the turn addresses a step the student never wrote" }
  ]
}
```

### 7.3 The transform, exactly

```ts
// lib/export.ts
export function buildLabelFiles(rows: ExportRow[], metrics: Map<string, Metric>) {
  const out = new Map<string, Map<string, Record<string, unknown>>>();  // handle → turnId → rec

  for (const r of rows) {
    if (r.gold_item_id !== null) continue;              // checks are never exported
    const m = metrics.get(r.metric_key)!;

    // Bucket. `labels/` must contain ONLY things fit_head may average.
    const bucket =
      r.retest_of !== null            ? 'retest'
      : m.scale !== 'ordinal'         ? 'covariates'
      : m.group === 'covariate'       ? 'covariates'
      : 'labels';

    const file = `${bucket}/${r.handle}`;
    const recs = out.get(file) ?? out.set(file, new Map()).get(file)!;
    const rec  = recs.get(r.turn_id) ?? { id: r.turn_id };
    recs.set(r.turn_id, rec);

    // NULL IS NOT ZERO. A not-applicable answer omits the key entirely.
    if (r.value !== null) rec[r.metric_key] = r.value;

    // `flag` is one string per turn in this format but one per (turn, metric) in the app,
    // so the metric key is carried inline and flags.jsonl holds the unlossy version.
    if (r.flag_kind) {
      const s = `${r.metric_key}: ${r.flag_kind}${r.flag_note ? ` — ${r.flag_note}` : ''}`;
      rec.flag = rec.flag ? `${rec.flag} | ${s}` : s;
    }
  }
  return out;
}
```

Rules, restated as assertions the exporter should check and the manifest should report:

1. `labels/` contains only `scale = 'ordinal'`, non-covariate metrics. `step_size` and any other
   non-ordinal key must never appear there — `fit_head` would take a mean of it, and a mean of
   2.0 on `length_fit` could be every turn correct or half of them cut off and half padded.
2. Values are JSON integers. Never `null`, never `0` for a not-applicable answer, and — for the
   binary guards especially — **never `true`/`false`**: `isinstance(True, int)` is `True` in
   Python, so booleans pass `fit_head`'s type check and are then averaged as 1 and 0, silently
   putting a guard on a different scale from the one its κ was computed on.
3. Retests live in `retest/`, because `agreement.load`'s last-write-wins would otherwise silently
   replace the first label with the second.
4. `units.json` contains only turns with at least one exported label, so `extract_hidden.py` does
   not spend GPU time on turns nobody rated.
5. `id` is identical in `units.json` and every label file. `extract_hidden` writes those ids into
   the npz and `fit_head` joins on them.
6. `shots` is `{}` for human raters. It exists so agent files can declare their few-shot units and
   `agreement.against_reference` can exclude them.

### 7.4 The commands that then work

```bash
python scripts/unpack_export.py export.json --into data/

python -m projects.pedagogy_rm.agreement \
    --labels 'data/labels/*.json' \
    --dimensions locates_student_object,diagnoses_the_error,applies_principle_here

python -m projects.pedagogy_rm.extract_hidden \
    --units data/label_slices/units.json --out data/hidden.npz

python -m projects.pedagogy_rm.fit_head \
    --hidden data/hidden.npz \
    --labels 'data/labels/*.json' \
    --slices  data/label_slices/units.json \
    --dimensions locates_student_object,diagnoses_the_error,applies_principle_here \
    --decouple '' \
    --cells 'locates_student_object=mean:16,diagnoses_the_error=last:16,applies_principle_here=mean:16'
```

**Three compatibility notes that will bite otherwise, and are about `rubric.py` rather than the
export:**

- Both scripts resolve keys through `rubric.BY_KEY`, so **`tutor_metrics/metrics.py` must register
  its metrics there** — same `Dimension` dataclass, appended to `BY_KEY`. Without it,
  `--dimensions asks_next_step` raises `KeyError`.
- `fit_head.CELLS` has no entry for any new key, so **`--cells` is mandatory** for every metric
  being fitted. It is not optional and the failure is a `KeyError`, not a warning.
- `fit_head` defaults to `--decouple leak`, which will load `--slices` and then do nothing.
  Pass `--decouple ''` or the metric you actually mean.

### 7.5 One source of truth for the rubric

`metrics.py` is the source of truth; the `metric` table is a cache. `scripts/seed_metrics.ts`
pushes the definitions with a `source_sha`, and `/api/health` fails if any row's `source_sha`
differs from the seeded manifest. This exists because the repo has already been burned by exactly
this drift: a run rated six dimensions against a rubric describing five, two dimensions were
requested without being defined, and one came back `2` on all 25 holdout turns, correlating −0.36
with the human. An undefined dimension does not fail loudly.

---

## 8. Deployment

Boring throughout. One project, one database, no third service.

### 8.1 Primitives

| Concern | Choice | Why not something else |
|---|---|---|
| App | Next.js App Router, TypeScript, **Node runtime** | Edge has compatibility problems and buys nothing here; Fluid Compute is the default and runs plain Node. |
| Database | **Neon Postgres via Vercel Marketplace** — `vercel integration add neon` | Vercel Postgres no longer exists as a first-party product; Neon is its Marketplace successor. Auto-provisions the env vars. |
| Driver | `@neondatabase/serverless`, `Pool`, one module-level instance | Supports real transactions (the assignment route needs one). `pg` would also work. **No ORM** — about fifteen queries, several of which are hand-tuned `FOR UPDATE SKIP LOCKED`, and an ORM would obscure exactly the ones that matter. |
| Migrations | Numbered `.sql` + `scripts/migrate.ts` + a `schema_migrations` table | No Prisma, no drizzle-kit, no generated client. |
| Sessions | Opaque token in an HttpOnly cookie, `sha256` in Postgres | No JWT library, no signing key to rotate, revocable in one `UPDATE`. |
| Bots | `botid` | Free at Basic on all plans; Deep Analysis on two routes only (§6.3). |
| Rate limits | Vercel WAF custom rules + per-rater SQL | No Redis. Upstash would be a fourth service for something two `count(*)`s already do. |
| Scheduled work | Vercel Cron → route handlers | 300 s default timeout is ample for the nightly QC pass. |
| Export artifacts | Vercel Blob, `access: 'private'` | Keeps large snapshots out of the response path. |
| Config | `vercel.ts` with `@vercel/config` | Replaces `vercel.json`. |

```ts
// vercel.ts
import type { VercelConfig } from '@vercel/config/v1';

export const config: VercelConfig = {
  framework: 'nextjs',
  crons: [
    { path: '/api/cron/qc',       schedule: '0 3 * * *' },
    { path: '/api/cron/snapshot', schedule: '30 3 * * *' },
  ],
};
```

```ts
// lib/db.ts — lazy, because top-level module code runs at build time and
// DATABASE_URL is not set on the very first deploy.
import { Pool } from '@neondatabase/serverless';

let pool: Pool | null = null;
export function db(): Pool {
  if (!pool) pool = new Pool({ connectionString: process.env.DATABASE_URL! });
  return pool;
}

export async function withTx<T>(fn: (c: PoolClient) => Promise<T>): Promise<T> {
  const c = await db().connect();
  try {
    await c.query('begin');
    const out = await fn(c);
    await c.query('commit');
    return out;
  } catch (e) { await c.query('rollback'); throw e; }
  finally { c.release(); }
}
```

Do **not** wrap the client in a JavaScript `Proxy` for lazy init — it breaks libraries that
introspect the object and fails with a hang rather than an error.

### 8.2 Environment variables

| Name | Set by | Used for |
|---|---|---|
| `DATABASE_URL` | Neon integration | Pooled connection string. |
| `DATABASE_URL_UNPOOLED` | Neon integration | Migrations only. |
| `ADMIN_TOKEN` | manual | Bearer token for `/api/admin/*`. 32 random bytes. |
| `CRON_SECRET` | manual | Bearer token for `/api/cron/*`. |
| `IP_HASH_PEPPER` | manual | Salts the IP hash so it cannot be reversed by enumeration. |
| `BLOB_READ_WRITE_TOKEN` | Blob store | Export snapshots. |
| `ACTIVE_BATCH_ID` | manual | Which batch is being served. One string, changed to cut over. |
| `GUIDE_VERSION` | manual | Stamped on every label. Bump when the rules change. |
| `NEXT_PUBLIC_UI_VERSION` | build | Stamped on every label; lets a UI regression be isolated later. |

### 8.3 Provisioning, in order

```bash
vercel link
vercel integration add neon            # provisions the DB, injects DATABASE_URL

for v in ADMIN_TOKEN CRON_SECRET IP_HASH_PEPPER ACTIVE_BATCH_ID GUIDE_VERSION; do
  vercel env add "$v" production        # one at a time; reads the value from stdin
done
vercel env pull .env.local --yes

npx dotenv -e .env.local -- npx tsx scripts/migrate.ts        # 001_init.sql …
npx dotenv -e .env.local -- npx tsx scripts/seed_metrics.ts   # from tutor_metrics/metrics.py
npx dotenv -e .env.local -- npx tsx scripts/ingest_turns.ts \
    --pool  data/pool.json  --gold data/gold.json --key data/key.json
# key.json STAYS LOCAL. Nothing in it is uploaded.

vercel deploy --prod
# then, staged, one at a time, reviewing traffic between each:
vercel firewall rules add … --action log --yes && vercel firewall publish --yes
```

`dotenv -e .env.local` is required: only Next.js auto-loads `.env.local`, and `tsx` scripts will
otherwise see no `DATABASE_URL`.

### 8.4 Build order, with gates

Mirroring `PLAN.md` §6 — each phase has a condition before the next is worth starting.

| # | Phase | Gate |
|---|---|---|
| 1 | Migrations + `seed_metrics` + `ingest_turns` | Every active metric has anchors, guidance, and a `source_sha`; `/api/health` green |
| 2 | Session, landing, `/api/session`, `/api/me` | A fresh browser gets a handle and keeps it across a reload |
| 3 | The labelling screen against a hard-coded assignment | The researcher labels 50 turns herself and finds the keyboard flow faster than `label_ui.py`. If it is not, fix it before anyone else sees it. |
| 4 | Claim query, `/api/assignment`, `/api/label` | Two browsers labelling at once never receive the same item; `item.n_labels` matches `count(label)` exactly |
| 5 | Calibration + primer + gold injection | The researcher and one other person both pass; the quiz teaches at least one thing each got wrong |
| 6 | Export + `unpack_export.py` | `agreement.py` and `fit_head.py` run on real exported data with **no edits to either file** |
| 7 | QC cron + admin dashboard | A deliberately bad rater (answer `2` to everything, 300 ms each) lands on probation within one nightly pass |
| 8 | WAF rules, BotID, spend limit | Staged in log mode, traffic reviewed, then enforced |
| 9 | Open the link | — |

Phase 6 before phase 9 is the one that is tempting to reorder and must not be. An export format
discovered to be wrong after 5,000 labels is a migration; discovered after 50, it is an
afternoon.

### 8.5 Cost

Neon's smallest paid tier and Vercel Pro. The database is tiny — 40,000 turns and 200,000 labels
is comfortably under a gigabyte. The only usage-linked line is BotID Deep Analysis at roughly $1
per 12,000 labels (§6.3). Set a spend limit anyway; blocked WAF traffic is not billed.

---

## 9. Where I guessed

| # | Guess | Why it matters | Cheapest way to settle it |
|---|---|---|---|
| 1 | **Guards are binary on `lo=1, hi=2`** (1 fails the gate, 2 passes). `PLAN.md` §2 gives guard questions but no scale. | `agreement.weighted_kappa` and `fit_head` both read `dim.lo`/`dim.hi`, so this works, but a 1–2 scale makes `fit_head`'s `spread < 0.15` check proportionally stricter. | Decide it in `metrics.py`. |
| 2 | **`assistance_level` is ordinal 1–6, `step_size` is non-ordinal 1–3, `student_state` is ordinal 1–3.** Graesser's ladder is ordered; `step_size` is explicitly "2 is good, not 3". | Determines whether κ is weighted and whether anything may average them. All three are covariates, so none reaches `labels/`. | Same. |
| 3 | **Overlap is 20% of (turn, metric) *items*, not of turns.** `PLAN.md` §6 says "20% double-labelled" about turns. | κ is computed per metric, so the item is the right unit; at the turn level, a doubled turn could still have every metric singly labelled. | Confirm the intent; the schema supports either. |
| 4 | **The 15-item quiz passes at 70% with a `says_only` override.** No pass mark is specified. | Too strict loses volunteers; too loose defeats the point. | Run the quiz on five known-good raters before opening the link and read the score distribution. |
| 5 | **The 2-item per-metric primer is an addition.** Not in `PLAN.md`. | It is where most of the 3–5× protocol effect lives, and it is cheap. | Ship it; drop it if raters complain. |
| 6 | **Intra-rater retest is included opportunistically.** `PLAN.md` §5 does not mention it; `LABELING_GUIDE.md` §4 treats it as non-optional. | It only yields data from raters returning after 72 h, which on a public link is a minority. Report the *n*. | Nothing to settle; just report honestly. |
| 7 | **Anonymous session, no signup.** | The whole conversion funnel depends on this. | If a paid platform (Prolific etc.) is used instead, most of §6.1 becomes irrelevant and a completion-code route is needed. |
| 8 | **Admin auth is a bearer token, not hosted auth.** A deliberate departure from the stated constraint. | One admin, no credential storage. Clerk is more parts, not fewer. | Say the word and it's a Clerk drop-in. |
| 9 | **No IRB / human-subjects process applies.** | Consent cannot be retro-fitted onto labels already collected. | Ask, before launch, not after. §6.6. |
| 10 | **Reference solutions exist for every question by seed time.** `build_pilot.py` generates them only for the pilot. | `no_reference_conflict` and `step_size` are undefined without one. The batch query already excludes turns with no reference from `needs_reference` metrics, so the failure is quiet: those metrics simply get fewer items. | Check the count in the manifest after materialising a batch. |
| 11 | **Inline LaTeX is rendered as plain text.** The corpus contains `\( \$3.45 \)`-style markup. | Real readability cost on a maths corpus, but KaTeX is a dependency and a rendering risk on model output. | Ship plain; add KaTeX behind a flag if raters say the maths is hard to read. |
| 12 | **The app lives at `projects/tutor_metrics/app/`** as a Vercel project with a root directory set. | Only affects the deploy config. | A separate repo is equally fine. |
| 13 | **`PLAN.md` §6 phase 6's gate has been met** — two agents reaching κ ≥ 0.5 on the surviving metrics — before the app opens. | Building a labelling app for metrics that fail their agreement gate spends volunteer effort on questions nobody can answer the same way. | Run the gate first. |

---

## 10. The two things most likely to go wrong

**The public link lands in the κ 0.13–0.30 regime anyway.** Everything in §2 and §3 is aimed at
this and none of it is a guarantee. The detection is built in: `metric_qc.kappa_w` is visible
nightly from the first week. The decision rule should be set now, before the data exists — if
pooled κ on a metric is below 0.4 after 200 pairs, the metric is rewritten or dropped, and no
volume of additional labelling substitutes. A probe cannot beat the noise in its own labels.

**Length reappears as the thing everyone is scoring.** `PLAN.md` exists because five dimensions
collapsed into one and that one was brevity. `metric_qc.length_rho` is on the dashboard for this
reason, computed per metric on **real turns only** — the V2 lesson is that a set padded with
synthetic negatives reported −0.37 where real turns alone gave −0.62. The exporter should stamp
`is_synthetic` into the manifest so that correlation can never accidentally be computed over the
padded pool.
