/**
 * Run the schema and every query that matters against an in-process Postgres.
 *
 *   npx tsx scripts/selftest.ts
 *
 * No database, no network, no credentials. It exists because the interesting parts of this app
 * are SQL - a partial unique index, three triggers, and a FOR UPDATE SKIP LOCKED claim - and a
 * typecheck says nothing about any of them. Run it before provisioning anything.
 */
import fs from 'node:fs';
import path from 'node:path';
import { PGlite } from '@electric-sql/pglite';
import { pgcrypto } from '@electric-sql/pglite/contrib/pgcrypto';
import { splitSentences, wordCount } from '../lib/sentences';
import { buildBundle, type ExportRow, type ExportUnitRow } from '../lib/export';
import { exportBucket, type MetricRow } from '../lib/metrics';
import { weightedKappa, runsZ, spearman, modeShare } from '../lib/qc';
import type { MetricSeedFile } from '../lib/metrics';

let failures = 0;

function check(name: string, condition: boolean, detail = '') {
  if (condition) {
    console.log(`  ok   ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
  }
}

async function refusal(fn: () => Promise<unknown>): Promise<string> {
  try {
    await fn();
    return '';
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  }
}

function eq(name: string, got: unknown, want: unknown) {
  check(name, JSON.stringify(got) === JSON.stringify(want), `got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
}

async function main() {
  const db = new PGlite({ extensions: { pgcrypto } });
  const sql = <T,>(text: string, params: unknown[] = []) => db.query<T>(text, params);

  console.log('\nschema');
  const ddl = fs.readFileSync(
    path.join(process.cwd(), 'db', 'migrations', '001_init.sql'),
    'utf8',
  );
  await db.exec(ddl);
  check('001_init.sql applies', true);

  const rel = await sql<{ n: number }>(
    `select count(*)::int as n from information_schema.tables where table_schema = 'public'`,
  );
  check('tables created', (rel.rows[0]?.n ?? 0) >= 15, `${rel.rows[0]?.n} relations`);

  console.log('\nmetric seed');
  const seed = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), 'db', 'metrics.json'), 'utf8'),
  ) as MetricSeedFile;
  for (const m of seed.metrics) {
    await sql(
      `insert into metric (key, version, grp, question, anchors, guidance, scale, lo, hi,
                           requires_span, span_from_value, needs_reference, needs_dialogue,
                           allow_na, source_sha)
       values ($1,1,$2::metric_group,$3,$4::jsonb,$5,$6::metric_scale,$7,$8,$9,$10,$11,$12,$13,$14)`,
      [
        m.key,
        m.group,
        m.question,
        JSON.stringify(m.anchors),
        m.guidance,
        m.scale,
        m.lo,
        m.hi,
        m.requiresSpan,
        m.spanFromValue,
        m.needsReference,
        m.needsDialogue,
        m.allowNa,
        m.sourceSha,
      ],
    );
  }
  const seeded = await sql<{ n: number }>('select count(*)::int as n from metric');
  eq('metrics seeded', seeded.rows[0]?.n, seed.metrics.length);

  const revisions = await sql<{ n: number }>('select count(*)::int as n from metric_revision');
  eq('metric_snapshot trigger recorded every wording', revisions.rows[0]?.n, seed.metrics.length);

  let refused = false;
  try {
    await sql(`update metric set question = 'changed?' where key = 'empty_praise'`);
  } catch {
    refused = true;
  }
  check('wording change without a version bump is refused', refused);

  for (const m of seed.metrics) {
    check(
      `${m.key}: anchors cover lo..hi`,
      m.anchors.length === m.hi - m.lo + 1 &&
        m.anchors.every((a, i) => a.value === m.lo + i),
      JSON.stringify(m.anchors.map((a) => a.value)),
    );
  }

  console.log('\ncontent + pools');
  await sql(
    `insert into dialogue (id, question, gold, reference, subject)
     values ('d1', 'Paul found 7 cents…', '$3.09', '7 + 98 = 105 cents…', 'math')`,
  );
  const turnText =
    'Good, the sum is right. Now look at the units — you wrote 309 with no dollar sign.\nWhat should it be?';
  await sql(
    `insert into turn (id, dialogue_id, pool, turn_index, student_before, tutor_turn, n_words)
     values ('t1','d1','production',5,'I added 7 + 98 + 204 and got 309.',$1,$2),
            ('t2','d1','production',5,'I do not know where to start.','Try again.',2),
            ('g1','d1','gold',5,'I do not know where to start.','What is 7 + 98?',5)`,
    [turnText, wordCount(turnText)],
  );

  await sql(`insert into batch (id, overlap_target, active) values ('b1', 1.0, true)`);
  // The message is asserted, not just the throw: the first version of this trigger raised
  // `record "new" has no field "source"` here, which looks exactly like a pass from the outside.
  const poolRefused = await refusal(
    () =>
      sql(`insert into item (batch_id, turn_id, metric_key, target_labels)
           values ('b1','g1','empty_praise',1)`),
  );
  check(
    'assert_pool blocks a gold turn becoming a production item',
    poolRefused.includes('cannot be a production item'),
    poolRefused,
  );

  const goldRefused = await refusal(
    () =>
      sql(`insert into gold_item (turn_id, metric_key, gold_value, rationale, role)
           values ('t1','empty_praise',1,'no','control')`),
  );
  check(
    'assert_pool blocks an authored answer on a production turn',
    goldRefused.includes('cannot hold an authored gold answer'),
    goldRefused,
  );

  await sql(
    `insert into gold_item (turn_id, metric_key, gold_value, rationale, role, use_in_quiz)
     values ('g1','locates_student_object',1,'Names nothing the student wrote.','hard_negative',true)`,
  );

  console.log('\nturn_public');
  const pub = await sql<{ id: string; question: string; reference: string }>(
    `select * from turn_public where id = 't1'`,
  );
  check('turn_public joins the dialogue', pub.rows[0]?.question.startsWith('Paul'));

  console.log('\nsentence split');
  const sentences = splitSentences(turnText);
  eq('three sentences', sentences.length, 3);
  check(
    'offsets address the original string',
    sentences.every((s) => turnText.slice(s.start, s.end) === s.text),
    JSON.stringify(sentences),
  );
  check('the $3.09 style decimal does not split', splitSentences('It is 3.09 dollars.').length === 1);

  console.log('\nthe claim query');
  await sql(
    `insert into item (batch_id, turn_id, metric_key, target_labels, shuffle)
     select 'b1', t.id, m.key, 2, random()
       from turn t cross join metric m
      where t.pool = 'production' and m.key in ('locates_student_object','assistance_level')`,
  );
  const items = await sql<{ n: number }>('select count(*)::int as n from item');
  eq('items materialised', items.rows[0]?.n, 4);

  const raterA = (
    await sql<{ id: string }>(`insert into rater (handle, status) values ('r_aaa','active') returning id`)
  ).rows[0].id;
  const raterB = (
    await sql<{ id: string }>(`insert into rater (handle, status) values ('r_bbb','active') returning id`)
  ).rows[0].id;
  const sessionA = (
    await sql<{ id: string }>(
      `insert into rater_session (rater_id, token_hash) values ($1, '\\x01'::bytea) returning id`,
      [raterA],
    )
  ).rows[0].id;
  const sessionB = (
    await sql<{ id: string }>(
      `insert into rater_session (rater_id, token_hash) values ($1, '\\x02'::bytea) returning id`,
      [raterB],
    )
  ).rows[0].id;

  const CLAIM = `with picked as (
      select i.id from item i
       where i.batch_id = $1 and i.metric_key = $2
         and i.n_labels < i.target_labels
         and (i.claimed_until is null or i.claimed_until < now())
         and not exists (select 1 from label l where l.item_id = i.id and l.rater_id = $3)
       order by (i.n_labels > 0) desc, i.priority, i.shuffle
       limit $4 for update skip locked)
    update item set claimed_until = now() + interval '45 minutes', claimed_by = $3
     where id in (select id from picked) returning id, turn_id`;

  const claimA = await sql<{ id: number; turn_id: string }>(CLAIM, [
    'b1',
    'locates_student_object',
    raterA,
    12,
  ]);
  eq('rater A claims both items', claimA.rows.length, 2);

  console.log('\nlabels and counters');
  await sql(
    `insert into rater_metric (rater_id, metric_key) values ($1,'locates_student_object')`,
    [raterA],
  );
  const assignmentA = (
    await sql<{ id: string }>(
      `insert into assignment (rater_id, session_id, batch_id, metric_key, kind, n_items)
       values ($1,$2,'b1','locates_student_object','production',2) returning id`,
      [raterA, sessionA],
    )
  ).rows[0].id;
  const aiA = await sql<{ presentation_id: string; item_id: number }>(
    `insert into assignment_item (assignment_id, position, item_id, served_at)
     select $1, i.ord, i.item_id, now() - interval '4 seconds'
       from jsonb_to_recordset($2::jsonb) as i(ord int, item_id bigint)
     returning presentation_id, item_id`,
    [assignmentA, JSON.stringify(claimA.rows.map((r, i) => ({ ord: i, item_id: r.id })))],
  );
  eq('assignment_item rows', aiA.rows.length, 2);

  const insertLabel = async (opts: {
    rater: string;
    session: string;
    assignment: string;
    presentation: string;
    turn: string;
    item: number | null;
    goldItem?: number | null;
    value: number | null;
    na?: string | null;
    span?: boolean;
    flag?: string | null;
  }) =>
    sql(
      `insert into label (rater_id, session_id, assignment_id, presentation_id, turn_id,
                          metric_key, metric_version, item_id, gold_item_id, value, na_reason,
                          span_start, span_end, span_text, sentence_index, flag_kind,
                          shown_at, dwell_ms, revisions,
                          metric_source_sha, guide_version, ui_version)
       values ($1,$2,$3,$4,$5,'locates_student_object',1,$6,$7,$8,$9,
               $10,$11,$12,$13,$14, now() - interval '4 seconds', 4000, 0, 'sha','g1','ui1')`,
      [
        opts.rater,
        opts.session,
        opts.assignment,
        opts.presentation,
        opts.turn,
        opts.item,
        opts.goldItem ?? null,
        opts.value,
        opts.na ?? null,
        opts.span ? 0 : null,
        opts.span ? 26 : null,
        opts.span ? turnText.slice(0, 26) : null,
        opts.span ? 0 : null,
        opts.flag ?? null,
      ],
    );

  const turnOf = new Map(claimA.rows.map((r) => [r.id, r.turn_id]));
  await insertLabel({
    rater: raterA,
    session: sessionA,
    assignment: assignmentA,
    presentation: aiA.rows[0].presentation_id,
    turn: turnOf.get(aiA.rows[0].item_id) as string,
    item: aiA.rows[0].item_id,
    value: 2,
    span: true,
  });

  const counters = await sql<{
    item_n: number;
    rater_n: number;
    session_n: number;
    assignment_n: number;
    metric_n: number;
    answered: number;
  }>(
    `select (select n_labels from item where id = $1)::int                                as item_n,
            (select n_labels from rater where id = $2)::int                               as rater_n,
            (select n_labels from rater_session where id = $3)::int                       as session_n,
            (select n_done from assignment where id = $4)::int                            as assignment_n,
            (select n_labels from rater_metric
              where rater_id = $2 and metric_key = 'locates_student_object')::int          as metric_n,
            (select count(*) from assignment_item
              where assignment_id = $4 and answered_at is not null)::int                   as answered`,
    [aiA.rows[0].item_id, raterA, sessionA, assignmentA],
  );
  eq('label_counters: item', counters.rows[0]?.item_n, 1);
  eq('label_counters: rater', counters.rows[0]?.rater_n, 1);
  eq('label_counters: session', counters.rows[0]?.session_n, 1);
  eq('label_counters: assignment', counters.rows[0]?.assignment_n, 1);
  eq('label_counters: rater_metric', counters.rows[0]?.metric_n, 1);
  eq('label_counters: assignment_item.answered_at', counters.rows[0]?.answered, 1);

  console.log('\nthe rules the schema is supposed to enforce');
  let dupRefused = false;
  const aiDup = (
    await sql<{ presentation_id: string }>(
      `insert into assignment_item (assignment_id, position, item_id, served_at)
       values ($1, 9, $2, now()) returning presentation_id`,
      [assignmentA, aiA.rows[0].item_id],
    )
  ).rows[0].presentation_id;
  try {
    await insertLabel({
      rater: raterA,
      session: sessionA,
      assignment: assignmentA,
      presentation: aiDup,
      turn: turnOf.get(aiA.rows[0].item_id) as string,
      item: aiA.rows[0].item_id,
      value: 1,
      span: false,
    });
  } catch {
    dupRefused = true;
  }
  check('one rater cannot label the same item twice', dupRefused);

  let nullRefused = false;
  try {
    await sql(
      `insert into label (rater_id, session_id, assignment_id, presentation_id, turn_id,
                          metric_key, metric_version, item_id, value, na_reason,
                          shown_at, dwell_ms, metric_source_sha, guide_version, ui_version)
       values ($1,$2,$3,$4,'t1','locates_student_object',1,$5,null,null,now(),10,'s','g','u')`,
      [raterA, sessionA, assignmentA, aiDup, aiA.rows[0].item_id],
    );
  } catch {
    nullRefused = true;
  }
  check('a null value without a reason is refused (nulls are never zeros)', nullRefused);

  console.log('\noverlap: a partner sorts ahead of a fresh item');
  const stillClaimed = await sql<{ claimed_until: Date | null }>(
    'select claimed_until from item where id = $1',
    [aiA.rows[0].item_id],
  );
  check(
    'an answered item releases its claim immediately',
    stillClaimed.rows[0]?.claimed_until === null,
    String(stillClaimed.rows[0]?.claimed_until),
  );
  const claimB = await sql<{ id: number }>(CLAIM, ['b1', 'locates_student_object', raterB, 1]);
  eq('rater B is offered the started pair first', claimB.rows[0]?.id, aiA.rows[0].item_id);

  const assignmentB = (
    await sql<{ id: string }>(
      `insert into assignment (rater_id, session_id, batch_id, metric_key, kind, n_items)
       values ($1,$2,'b1','locates_student_object','production',1) returning id`,
      [raterB, sessionB],
    )
  ).rows[0].id;
  await sql(`insert into rater_metric (rater_id, metric_key) values ($1,'locates_student_object')`, [
    raterB,
  ]);
  const aiB = (
    await sql<{ presentation_id: string }>(
      `insert into assignment_item (assignment_id, position, item_id, served_at)
       values ($1, 0, $2, now()) returning presentation_id`,
      [assignmentB, claimB.rows[0].id],
    )
  ).rows[0].presentation_id;
  await insertLabel({
    rater: raterB,
    session: sessionB,
    assignment: assignmentB,
    presentation: aiB,
    turn: turnOf.get(claimB.rows[0].id) as string,
    item: claimB.rows[0].id,
    value: 1,
    span: false,
    flag: 'rubric_misfit',
  });

  const pairDone = await sql<{ n_labels: number; target_labels: number }>(
    'select n_labels, target_labels from item where id = $1',
    [claimB.rows[0].id],
  );
  eq('the pair is complete', pairDone.rows[0]?.n_labels, 2);

  // Nothing is left for B: the pair they just completed is full, and the other item is still
  // soft-locked to rater A, who claimed it and walked away.
  const claimC = await sql<{ id: number }>(CLAIM, ['b1', 'locates_student_object', raterB, 12]);
  eq('a completed item is never offered again', claimC.rows.length, 0);

  await sql(
    `update item set claimed_until = now() - interval '1 minute'
      where claimed_by = $1 and n_labels = 0`,
    [raterA],
  );
  const claimD = await sql<{ id: number }>(CLAIM, ['b1', 'locates_student_object', raterB, 12]);
  eq('an abandoned claim self-heals with no reaper process', claimD.rows.length, 1);
  check(
    'and what comes back is the untouched item, not the finished one',
    claimD.rows[0]?.id !== claimB.rows[0].id,
  );

  console.log('\ngold checks');
  const goldId = (await sql<{ id: number }>(`select id from gold_item limit 1`)).rows[0].id;
  const aiG = (
    await sql<{ presentation_id: string }>(
      `insert into assignment_item (assignment_id, position, gold_item_id, served_at)
       values ($1, 5, $2, now()) returning presentation_id`,
      [assignmentB, goldId],
    )
  ).rows[0].presentation_id;
  await insertLabel({
    rater: raterB,
    session: sessionB,
    assignment: assignmentB,
    presentation: aiG,
    turn: 'g1',
    item: null,
    goldItem: goldId,
    value: 1,
    span: false,
  });
  const goldCounters = await sql<{ seen: number; correct: number; served: number }>(
    `select (select n_gold_seen from rater where id = $1)::int    as seen,
            (select n_gold_correct from rater where id = $1)::int as correct,
            (select n_served from gold_item where id = $2)::int   as served`,
    [raterB, goldId],
  );
  eq('gold seen', goldCounters.rows[0]?.seen, 1);
  eq('gold scored correct', goldCounters.rows[0]?.correct, 1);
  eq('gold_item.n_served', goldCounters.rows[0]?.served, 1);

  console.log('\nundo reverses every counter');
  await sql(`delete from label where presentation_id = $1`, [aiG]);
  const afterUndo = await sql<{ seen: number; session_n: number }>(
    `select (select n_gold_seen from rater where id = $1)::int         as seen,
            (select n_labels from rater_session where id = $2)::int    as session_n`,
    [raterB, sessionB],
  );
  eq('gold seen back to zero', afterUndo.rows[0]?.seen, 0);
  eq('session count back to one', afterUndo.rows[0]?.session_n, 1);

  console.log('\nmetric_progress view');
  const progress = await sql<{
    metric_key: string;
    n_items: string;
    got: string;
    pairs_done: string;
  }>(`select * from metric_progress where metric_key = 'locates_student_object'`);
  eq('progress: got', Number(progress.rows[0]?.got), 2);
  eq('progress: pairs done', Number(progress.rows[0]?.pairs_done), 1);

  console.log('\npickMetric ordering');
  const picked = await sql<{ key: string }>(
    `select m.key from metric m
       join metric_progress p on p.metric_key = m.key and p.batch_id = 'b1'
      where m.active
        and exists (select 1 from item i
                     where i.batch_id = 'b1' and i.metric_key = m.key
                       and i.n_labels < i.target_labels
                       and (i.claimed_until is null or i.claimed_until < now())
                       and not exists (select 1 from label l
                                        where l.item_id = i.id and l.rater_id = $1)
                     limit 1)
      order by (p.got::numeric / nullif(p.target, 0)) asc, random()
      limit 1`,
    [raterA],
  );
  eq('the least complete metric is chosen', picked.rows[0]?.key, 'assistance_level');

  console.log('\nexport transform');
  const metricRows = (
    await sql<MetricRow>(
      `select key, version, grp, question, anchors, guidance, scale, lo, hi, requires_span,
              span_from_value, needs_reference, needs_dialogue, allow_na, source_sha, active
         from metric`,
    )
  ).rows;
  const metrics = new Map(metricRows.map((m) => [m.key, m]));
  eq('assistance_level exports to labels/', exportBucket(metrics.get('assistance_level')!, false), 'labels');
  eq('icap_class exports to covariates/', exportBucket(metrics.get('icap_class')!, false), 'covariates');
  eq('answer_withheld exports to covariates/', exportBucket(metrics.get('answer_withheld')!, false), 'covariates');
  eq('empty_praise exports to labels/', exportBucket(metrics.get('empty_praise')!, false), 'labels');

  const exportRows = (
    await sql<ExportRow>(
      `select r.handle, l.turn_id, l.metric_key, l.value, l.na_reason, l.flag_kind, l.flag_note,
              l.gold_item_id, l.retest_of, l.span_start, l.span_end, l.span_text,
              l.sentence_index, l.answered_at
         from label l join rater r on r.id = l.rater_id`,
    )
  ).rows;
  const unitRows = (
    await sql<ExportUnitRow>(
      `select distinct tp.id, tp.question, tp.choices, tp.gold, tp.reference, tp.student_before,
              tp.tutor_turn, tp.turn_index, tp.subject, tp.grade
         from turn_public tp join label l on l.turn_id = tp.id and l.gold_item_id is null`,
    )
  ).rows;

  const bundle = buildBundle({
    rows: exportRows,
    units: unitRows,
    metrics,
    guideVersion: 'g1',
    filters: {},
  });
  const manifest = bundle.manifest as { counts: Record<string, unknown>; violations: string[] };
  eq('no export assertions failed', manifest.violations, []);
  check('units.json is present', 'label_slices/units.json' in bundle.files);
  check('two rater files', ['labels/r_aaa.json', 'labels/r_bbb.json'].every((f) => f in bundle.files));

  const fileA = bundle.files['labels/r_aaa.json'] as {
    schema: string;
    rater: string;
    shots: unknown;
    labels: Array<Record<string, unknown>>;
  };
  eq('labels file schema', fileA.schema, 'pedagogy-rm/labels-v1');
  eq('shots is empty for a human', fileA.shots, {});
  eq('value is a plain integer', fileA.labels[0].locates_student_object, 2);
  check(
    'value is not a boolean',
    typeof fileA.labels[0].locates_student_object === 'number',
    typeof fileA.labels[0].locates_student_object,
  );
  const fileB = bundle.files['labels/r_bbb.json'] as { labels: Array<Record<string, unknown>> };
  check(
    'a flag is a string carrying its metric key',
    typeof fileB.labels[0].flag === 'string' &&
      (fileB.labels[0].flag as string).startsWith('locates_student_object: rubric_misfit'),
    String(fileB.labels[0].flag),
  );
  const units = bundle.files['label_slices/units.json'] as {
    shared: number;
    units: Array<Record<string, unknown>>;
  };
  eq('shared units', units.shared, 1);
  check(
    'every unit carries the four fields extract_hidden reads',
    units.units.every((u) => u.id && u.question && u.tutor_turn !== undefined && u.student_before !== undefined),
  );
  check('spans file written', 'spans/r_aaa.jsonl' in bundle.files);

  // A known-answer check must never reach the export: it would be the rater agreeing with an
  // answer key, scored as if it were an independent opinion about a production turn.
  const withCheck = buildBundle({
    rows: [
      ...exportRows,
      {
        ...exportRows[0],
        turn_id: 'g1',
        gold_item_id: 99,
        value: 2,
      },
    ],
    units: [...unitRows, { ...unitRows[0], id: 'g1' }],
    metrics,
    guideVersion: 'g1',
    filters: {},
  });
  const checkUnits = withCheck.files['label_slices/units.json'] as {
    units: Array<{ id: string }>;
  };
  const checkLabels = Object.entries(withCheck.files)
    .filter(([name]) => name.startsWith('labels/'))
    .flatMap(([, file]) => (file as { labels: Array<{ id: string }> }).labels);
  check(
    'a gold check is excluded from labels/ and from units.json',
    !checkUnits.units.some((u) => u.id === 'g1') && !checkLabels.some((l) => l.id === 'g1'),
  );
  eq('and it is counted as excluded in the manifest',
    (withCheck.manifest as { counts: { checks_excluded: number } }).counts.checks_excluded, 1);

  if (process.argv.includes('--emit')) {
    const out = process.argv[process.argv.indexOf('--emit') + 1] ?? 'export.json';
    fs.writeFileSync(out, JSON.stringify(bundle, null, 1));
    console.log(`\n  wrote ${out}`);
  }

  console.log('\nQC statistics');
  eq('kappa is 1 on perfect agreement', weightedKappa([1, 2, 1, 2], [1, 2, 1, 2], 1, 2), 1);
  check(
    'kappa is negative on inversion',
    (weightedKappa([1, 2, 1, 2], [2, 1, 2, 1], 1, 2) ?? 0) < 0,
  );
  eq('mode share', modeShare([1, 1, 1, 2]).share, 0.75);
  check('spearman is 1 on a monotone pair', (spearman([1, 2, 3, 4], [2, 4, 6, 8]) ?? 0) > 0.999);
  check(
    'runs_z is strongly negative on a clumped sequence',
    (runsZ([...Array(20).fill(1), ...Array(20).fill(2)], 1) ?? 0) < -3,
  );

  console.log(failures === 0 ? '\nall checks passed' : `\n${failures} CHECK(S) FAILED`);
  await db.close();
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
