# The tutor-metrics labelling app

A Next.js app on Vercel with one Neon Postgres behind it. It serves one `(turn, metric)` pair at a
time to anonymous raters, gates them behind a calibration quiz, and exports labels in the exact
shape `projects/pedagogy_rm/extract_hidden.py` and `fit_head.py` already read.

Built from `../LABELING_APP_SPEC.md`. The rubric comes from `../metrics.py` and nothing else —
every question, anchor and rung description on screen is copied out of that file verbatim.

---

## The one-minute version

```bash
npm install
python3 scripts/dump_metrics.py            # metrics.py -> db/metrics.json
npx tsx scripts/selftest.ts                # schema + queries + export, no database needed
npm run build
```

`selftest.ts` runs the whole schema and the queries that matter against an in-process Postgres. It
needs no credentials and takes two seconds. Run it before provisioning anything.

---

## What is here

```
app/
  page.tsx              landing, consent, one button
  how/page.tsx          the four rules, one screen
  calibration/page.tsx  the quiz, feedback after every item
  label/page.tsx        THE PRODUCT
  me/page.tsx           the rater's own progress
  admin/page.tsx        dashboard, bearer-token gated
  api/…/route.ts        19 routes, Node runtime, zod on every input
components/
  LabelStage.tsx        the labelling screen: keyboard, phases, span selection
  AnswerPanel.tsx       binary/ordinal anchors, the assistance_level ladder, span and n/a prompts
  ReadingColumn.tsx     problem, reference, dialogue, student message, the tutor turn
  Labeller.tsx          production loop: fetch a set, hold it, POST each answer
  CalibrationRunner.tsx quiz loop with immediate feedback
  Overlays.tsx          the `?` anchors sheet and the flag picker
  AdminDashboard.tsx  MyProgress.tsx  StartButton.tsx  ThemeToggle.tsx
lib/
  db.ts        pool + withTx          session.ts   cookie issue and lookup
  metrics.ts   the metric cache        assign.ts    the claim query and hydration
  label.ts     the one insert          calibration.ts  quiz build and scoring
  export.ts    the §7 transform        qc.ts        kappa, spearman, runs, mode share
  auth.ts  config.ts  sentences.ts  state.ts  types.ts  validate.ts
db/
  migrations/001_init.sql   the whole schema
  metrics.json              generated from metrics.py; the seed and the drift check read it
scripts/
  dump_metrics.py   metrics.py -> db/metrics.json
  migrate.ts        apply db/migrations/*.sql once each
  seed_metrics.ts   push db/metrics.json into the metric table
  ingest_turns.ts   generate.py moments -> blinded rows, key.json stays local
  selftest.ts       schema, triggers, claim query, export, statistics
  unpack_export.py  export bundle -> data/ in the layout the analysis scripts read
```

---

## Deploying

### 1. Provision

```bash
cd projects/tutor_metrics/app
vercel link                              # root directory = projects/tutor_metrics/app
vercel integration add neon              # injects DATABASE_URL and DATABASE_URL_UNPOOLED
```

### 2. Environment variables

| Name | Set by | Used for |
|---|---|---|
| `DATABASE_URL` | Neon integration | pooled connection, the app |
| `DATABASE_URL_UNPOOLED` | Neon integration | migrations and bulk loads |
| `ADMIN_TOKEN` | you | `Authorization: Bearer` for `/api/admin/*` |
| `CRON_SECRET` | you | `Authorization: Bearer` for `/api/cron/*`; Vercel Cron sends it |
| `IP_HASH_PEPPER` | you | salts the IP hash. No raw IP is ever stored |
| `ACTIVE_BATCH_ID` | you | which batch is served. One string, changed to cut over |
| `GUIDE_VERSION` | you | stamped on every label; bump when the rules change |
| `NEXT_PUBLIC_UI_VERSION` | you | stamped on every label; isolates a UI regression later |
| `BLOB_READ_WRITE_TOKEN` | Blob store | optional. Without it the snapshot cron stores nothing and says so |
| `QUIZ_SIZE`, `QUIZ_PASS_MARK` | optional | default 15 and 10.5 |

```bash
for v in ADMIN_TOKEN CRON_SECRET IP_HASH_PEPPER ACTIVE_BATCH_ID GUIDE_VERSION NEXT_PUBLIC_UI_VERSION; do
  vercel env add "$v" production
done
vercel env pull .env.local --yes
```

`openssl rand -hex 32` for the three secrets.

### 3. Schema and rubric

`dotenv -e .env.local` is required: only Next.js auto-loads `.env.local`, and a `tsx` script would
otherwise see no `DATABASE_URL`.

```bash
python3 scripts/dump_metrics.py
npx dotenv -e .env.local -- npx tsx scripts/migrate.ts
npx dotenv -e .env.local -- npx tsx scripts/seed_metrics.ts
```

### 4. Turns

`generate.py` writes one moment per line. `ingest_turns.ts` salts every id locally, uploads only
blind-safe fields, and writes `key.json`, which maps the salted ids back to model, style, scenario
and — the one that would poison the project — the scenario's prescribed `target_level`.

**`key.json` never leaves your laptop.** `.gitignore` already excludes it. The database cannot leak
provenance because it never receives any.

```bash
npx dotenv -e .env.local -- npx tsx scripts/ingest_turns.ts \
    --moments ../../../data/tutor_metrics/moments.jsonl \
    --key data/key.json \
    --batch b1 --overlap 0.2 --activate \
    --metrics locates_student_object,assistance_level,diagnoses_the_error
```

`--dry-run` writes the key and reports the counts without touching the database. Truncated turns
are skipped unless `--include-truncated`: a rater scoring a turn the cap cut off is scoring the cap.

Reuse the same `--key` file on later runs. The salt is read back out of it, so the same content
keeps the same id; a fresh salt would orphan every label already collected.

### 5. Calibration content

The quiz is authored, not generated. Ingest a small pool with `--pool gold`, then post the answers:

```bash
npx dotenv -e .env.local -- npx tsx scripts/ingest_turns.ts \
    --moments data/gold_moments.jsonl --key data/goldkey.json --pool gold

curl -X POST "$APP/api/admin/gold" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H 'content-type: application/json' \
  -d '{"goldItems":[{"turnId":"t…","metricKey":"locates_student_object","goldValue":2,
       "rationale":"Quotes the student’s own line back at them.",
       "role":"positive_canonical","useInQuiz":true,"quizPosition":0}]}'
```

The roles the spec asks for, per 15-item quiz: 4 `positive_canonical`, 2 `positive_atypical`,
2 `positive_partial`, 4 `hard_negative`, 3 `says_only`. Author a second set of 15 so a retake draws
different items — the quiz query excludes anything the rater has already seen.

Gold items with `useInQuiz: false` are the pool the 2-item metric primer and the injected
production checks draw from. Author some, or the primer is silently skipped.

**Until quiz items exist, nobody can qualify.** To label the first fifty turns yourself, mark your
own rater active:

```bash
curl -X PATCH "$APP/api/admin/rater/$RATER_ID" \
  -H "Authorization: Bearer $ADMIN_TOKEN" -H 'content-type: application/json' \
  -d '{"status":"active","reason":"researcher, phase 3"}'
```

### 6. Deploy, then the firewall

```bash
vercel deploy --prod
curl "$APP/api/health"     # ok:true, metricShaOk:true
```

Stage every WAF rule in log mode and read the dashboard before enforcing. The label limit is
deliberately loose: a seminar room on one NAT is exactly the audience this app is for, and a limit
tight enough to catch one abusive rater would take out the whole room.

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

Set a Vercel spend limit. The only usage-linked line is BotID Deep Analysis, on `/api/session` and
`/api/assignment` only — roughly $1 per 12,000 labels. `/api/label` is not gated because a client
that cleared the assignment check already holds a valid session.

`vercel.json` registers the two crons: `/api/cron/qc` at 03:00 and `/api/cron/snapshot` at 03:30.

---

## Running locally

You need a Postgres. Anything works — Neon's own branch, Docker, or Homebrew:

```bash
docker run -d --name tutorlabel -e POSTGRES_PASSWORD=dev -p 5433:5432 postgres:16
export DATABASE_URL='postgres://postgres:dev@localhost:5433/postgres?sslmode=disable'

npx tsx scripts/migrate.ts
npx tsx scripts/seed_metrics.ts
npx tsx scripts/ingest_turns.ts --moments … --key data/key.json --batch b1 --overlap 0.2 --activate
npm run dev
```

`sslmode=disable` in the connection string turns off TLS; anything else gets TLS with
`rejectUnauthorized: false`, which is what Neon wants.

BotID returns `isBot: false` in development, so the protected routes are open locally. Testing them
with `curl` against production will be blocked; go through a `fetch` from the app.

---

## Export, and the analysis that consumes it

```bash
curl -sH "Authorization: Bearer $ADMIN_TOKEN" \
     "$APP/api/admin/export?batch=b1&min_status=active" > export.json
python3 scripts/unpack_export.py export.json --into data/
```

```
data/
  label_slices/units.json    {"schema","shared","units":[…]}      <- extract_hidden, --slices
  labels/<handle>.json       ordinal, non-covariate metrics only  <- fit_head, agreement
  covariates/<handle>.json   icap_class, answer_withheld          <- NEVER globbed into labels/
  retest/<handle>.json       second-pass labels only
  spans/<handle>.jsonl       the audit trail for the says-only failure mode
  flags.jsonl                the unlossy flag record
  manifest.json              counts, filters, metric shas, assertion violations
```

`min_status` is `active`, `probation` (the default, meaning active **and** probation) or `any`.
Labels are never deleted and status is never applied retroactively, so excluding a cohort is a
filter here rather than a change to the data — which means the effect of excluding them can be
measured instead of assumed.

`unpack_export.py` exits non-zero if the manifest reports any assertion violation. The assertions
are the ones that would otherwise fail silently: a non-ordinal key in `labels/`, a value that is
not a JSON integer, a labelled turn with no unit row.

### One prerequisite on the Python side

`agreement.py` and `fit_head.py` resolve keys through `pedagogy_rm.rubric.BY_KEY`, which does not
know the tutor-metrics keys, so `--dimensions locates_student_object` raises `KeyError`. Register
them once, in `rubric.py`:

```python
import json, pathlib
from projects.pedagogy_rm.rubric import BY_KEY, Dimension

_seed = json.loads(pathlib.Path("projects/tutor_metrics/app/db/metrics.json").read_text())
for _m in _seed["metrics"]:
    BY_KEY[_m["key"]] = Dimension(
        key=_m["key"],
        question=_m["question"],
        anchors={a["value"]: a["label"] for a in _m["anchors"]},
    )
```

Then, with no edits to either analysis script:

```bash
python -m projects.pedagogy_rm.agreement \
    --labels 'data/labels/*.json' \
    --dimensions locates_student_object,diagnoses_the_error,assistance_level

python -m projects.pedagogy_rm.extract_hidden \
    --units data/label_slices/units.json --out data/hidden.npz

python -m projects.pedagogy_rm.fit_head \
    --hidden data/hidden.npz \
    --labels 'data/labels/*.json' \
    --slices data/label_slices/units.json \
    --dimensions locates_student_object,diagnoses_the_error,assistance_level \
    --decouple '' \
    --cells 'locates_student_object=mean:16,diagnoses_the_error=last:16,assistance_level=mean:16'
```

`--cells` is mandatory — `fit_head.CELLS` has no entry for any new key and the failure is a
`KeyError`, not a warning. `--decouple ''` is needed because it defaults to `leak`, which does not
exist here.

---

## Operating it

`/admin` wants `ADMIN_TOKEN` pasted once; it is kept in `sessionStorage` for that tab only.

The nightly QC pass writes `rater_qc` and `metric_qc`, moves raters between `active`, `probation`
and `suspended`, and promotes any `(turn, metric)` where three or more qualified raters agreed
exactly to earned gold. Trigger it by hand with either bearer token:

```bash
curl -H "Authorization: Bearer $ADMIN_TOKEN" "$APP/api/cron/qc"
```

The dashboard's alerts are the guide's standing rules, wired in rather than left to be noticed:

- **flag rate > 20% on a metric → rewrite or drop it, whatever its kappa.** A high flag rate is a
  low kappa arriving more politely.
- kappa below 0.4 after 200 pairs → the metric is rewritten or dropped, and no volume of additional
  labelling substitutes.
- |length rho| > 0.6 → the pool is scoring length, which is the failure this whole project exists
  to fix.
- one value on more than 90% of labels → a prevalence problem more labelling will not fix.

Two things are deliberately not shown to raters: peer agreement, ever, while collection is open —
it turns the task from *read the anchors and answer* into *guess what the others said* — and
anything that identifies a check item.

---

## Things worth knowing before you change something

**Overlap is data, not code.** `item.target_labels` is set once at materialisation from
`batch.overlap_target`. Raising it on a shaky metric is an `UPDATE`, not a redeploy:

```sql
update item set target_labels = 3 where batch_id = 'b1' and metric_key = 'demand_is_specific';
```

**Nulls are never zeros.** A not-applicable answer stores `value = null` with a `na_reason` and the
export omits the key entirely. `fit_head` skips a missing key natively. Writing 0 would convert
"the question did not apply" into "the artifact did it badly".

**Changing a question's wording requires a version bump.** The `metric_snapshot` trigger refuses
otherwise, because a wording change retroactively reinterprets every label already collected.
`scripts/seed_metrics.ts` does the bump for you when `metrics.py` changes.

**`/api/health` fails when the seeded `source_sha` and `metrics.py` disagree.** That is the point:
this repo has already shipped a run that rated six dimensions against a rubric describing five.
