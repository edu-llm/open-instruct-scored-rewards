-- The labelling app's whole data model. LABELING_APP_SPEC.md §1.2.
--
-- Two rules the schema enforces structurally:
--   Provenance never reaches the server. Ids are salted by scripts/ingest_turns.ts and the key
--   stays on the researcher's laptop, so the database cannot leak model, style, temperature or
--   scenario target because it never receives them.
--   Nulls are never zeros. label.value is nullable, na_reason is non-null exactly when it is,
--   and the export omits the key rather than emitting a null.

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
  reference   text,                        -- worked solution
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
-- own answers measures the labeller's belief about the answer. Earned gold is derived
-- from production labels, so its turn legitimately stays in the production pool.
-- The branches are nested rather than two flat conditions on tg_table_name: PL/pgSQL plans a whole
-- boolean expression before evaluating it, so `tg_table_name = 'gold_item' and new.source = …`
-- fails with `record "new" has no field "source"` on every insert into `item`.
create or replace function assert_pool() returns trigger language plpgsql as $$
declare p turn_pool;
begin
  select pool into p from turn where id = new.turn_id;
  if tg_table_name = 'item' then
    if p <> 'production' then
      raise exception 'turn % is pool=%; cannot be a production item', new.turn_id, p;
    end if;
  else
    if new.source = 'authored' and p <> 'gold' then
      raise exception 'turn % is pool=%; cannot hold an authored gold answer', new.turn_id, p;
    end if;
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

  -- The claim is released the moment the label lands. Without this, a target_labels = 2 item is
  -- locked to its first rater for the full 45 minutes even though they have already answered it,
  -- and the partner - the only thing kappa can be computed from - arrives 45 minutes late.
  -- Correctness never depended on the lock; label_one_per_rater_idx does that.
  if r.item_id is not null and r.retest_of is null then
    update item set n_labels      = n_labels + d,
                    claimed_until = null,
                    claimed_by    = null
              where id = r.item_id;
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
