/**
 * Load moments from projects/tutor_metrics/generate.py into the database, blind.
 *
 *   npx dotenv -e .env.local -- npx tsx scripts/ingest_turns.ts \
 *       --moments ../../../data/tutor_metrics/moments.jsonl \
 *       --key data/key.json --batch b1 --overlap 0.2 --activate
 *
 * BLINDING IS A PROPERTY OF THE DEPLOYMENT, NOT OF THE UI. build_blind_set.py already splits a
 * pool from a key and keeps the key on disk; this does the same, one layer earlier. The salted id
 * is computed here, only blind-safe fields are uploaded, and key.json stays on the laptop. The
 * database therefore cannot leak the model, the style, the scenario or - the one that would
 * actually poison the project - the scenario's prescribed target level, because it never receives
 * any of them.
 *
 * `stratum` is the deliberate exception: an opaque hash of (scenario, style) so a batch can be
 * balanced across cells without the balancing telling anyone what the cells are.
 */
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { arg, client, flag } from './_pg';

interface Moment {
  item_id: string | number | null;
  question: string;
  solution: string | null;
  answer: string | null;
  subject: string | null;
  scenario: string;
  target_level: string;
  scenario_check?: string;
  student_text: string;
  model: string;
  tutor_turns: Array<{ style: string; text: string; truncated: boolean }>;
}

const WITHHELD = ['style', 'temperature', 'sample_index', 'probe', 'model', 'scenario', 'target_level'];

function digest(parts: string[], salt: string, n = 14): string {
  return crypto
    .createHash('sha256')
    .update(salt + parts.join('\u241f'))
    .digest('hex')
    .slice(0, n);
}

function words(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

async function main() {
  const momentsPath = arg('moments');
  if (!momentsPath) throw new Error('--moments <generate.py jsonl> is required');
  const keyPath = arg('key', 'data/key.json') as string;
  const pool = (arg('pool', 'production') as 'production' | 'gold') ?? 'production';
  const batchId = arg('batch');
  const overlap = Number(arg('overlap', '0.2'));
  const metrics = arg('metrics')?.split(',').filter(Boolean) ?? null;
  const limit = Number(arg('limit', '0'));
  const includeTruncated = flag('include-truncated');
  const dryRun = flag('dry-run');

  // Reused across runs so the same content keeps the same id; regenerating it would orphan every
  // label already collected.
  let salt = arg('salt');
  if (!salt && fs.existsSync(keyPath)) {
    salt = (JSON.parse(fs.readFileSync(keyPath, 'utf8')) as { salt?: string }).salt;
  }
  if (!salt) salt = crypto.randomBytes(16).toString('hex');

  const lines = fs
    .readFileSync(momentsPath, 'utf8')
    .split('\n')
    .filter((l) => l.trim());
  const moments = (limit ? lines.slice(0, limit) : lines).map((l) => JSON.parse(l) as Moment);

  const dialogues = new Map<string, Record<string, unknown>>();
  const turns: Array<Record<string, unknown>> = [];
  const key: Record<string, unknown> = {};
  let truncatedSkipped = 0;

  for (const m of moments) {
    const itemId = String(m.item_id ?? digest([m.question], salt));
    const dialogueId = `d${digest(['dialogue', itemId], salt)}`;
    if (!dialogues.has(dialogueId)) {
      dialogues.set(dialogueId, {
        id: dialogueId,
        subject: m.subject ?? null,
        grade: null,
        question: m.question,
        choices: null,
        gold: m.answer ?? null,
        reference: m.solution ?? null,
      });
    }

    for (const t of m.tutor_turns) {
      // A rater scoring a truncated turn is scoring the cap, not the tutor.
      if (t.truncated && !includeTruncated) {
        truncatedSkipped += 1;
        continue;
      }
      const content = digest([m.student_text, t.text], salt, 16);
      const turnId = `t${digest([itemId, m.scenario, t.style, content], salt)}`;
      turns.push({
        id: turnId,
        dialogue_id: dialogueId,
        pool,
        turn_index: 0,
        context: [],
        student_before: m.student_text,
        tutor_turn: t.text,
        n_words: words(t.text),
        stratum: `s${digest(['stratum', m.scenario, t.style], salt, 6)}`,
      });
      key[turnId] = {
        item_id: itemId,
        scenario: m.scenario,
        target_level: m.target_level,
        style: t.style,
        model: m.model,
        truncated: t.truncated,
        dialogue_id: dialogueId,
      };
    }
  }

  for (const row of turns) {
    for (const banned of WITHHELD) {
      if (banned in row) throw new Error(`refusing to upload ${banned} on turn ${row.id}`);
    }
  }

  console.log(
    `${moments.length} moments -> ${dialogues.size} dialogues, ${turns.length} turns` +
      (truncatedSkipped ? `, ${truncatedSkipped} truncated turns skipped` : ''),
  );

  fs.mkdirSync(path.dirname(path.resolve(keyPath)), { recursive: true });
  fs.writeFileSync(
    keyPath,
    JSON.stringify({ salt, generated_at: new Date().toISOString(), key }, null, 1),
  );
  console.log(`  key  ${keyPath}   <- STAYS LOCAL. Nothing in it is uploaded.`);

  if (dryRun) {
    console.log('  dry run — nothing written to the database');
    return;
  }

  const c = await client();
  try {
    await c.query('begin');
    for (const chunk of chunked([...dialogues.values()], 200)) {
      await c.query(
        `insert into dialogue (id, subject, grade, question, choices, gold, reference)
         select d.id, d.subject, d.grade, d.question, d.choices, d.gold, d.reference
           from jsonb_to_recordset($1::jsonb) as d(
                id text, subject text, grade int, question text,
                choices jsonb, gold text, reference text)
         on conflict (id) do nothing`,
        [JSON.stringify(chunk)],
      );
    }
    let insertedTurns = 0;
    for (const chunk of chunked(turns, 200)) {
      const res = await c.query(
        `insert into turn (id, dialogue_id, pool, turn_index, context,
                           student_before, tutor_turn, n_words, stratum)
         select t.id, t.dialogue_id, t.pool::turn_pool, t.turn_index, t.context,
                t.student_before, t.tutor_turn, t.n_words, t.stratum
           from jsonb_to_recordset($1::jsonb) as t(
                id text, dialogue_id text, pool text, turn_index int, context jsonb,
                student_before text, tutor_turn text, n_words int, stratum text)
         on conflict (id) do nothing
         returning id`,
        [JSON.stringify(chunk)],
      );
      insertedTurns += res.rowCount ?? 0;
    }
    console.log(`  inserted ${insertedTurns} new turns (${turns.length - insertedTurns} already present)`);

    if (batchId && pool === 'production') {
      await c.query(
        `insert into batch (id, overlap_target, active) values ($1, $2, $3)
         on conflict (id) do update set overlap_target = excluded.overlap_target,
                                        active = excluded.active`,
        [batchId, overlap, flag('activate')],
      );
      const res = await c.query(
        `insert into item (batch_id, turn_id, metric_key, target_labels, priority)
         select $1, t.id, m.key,
                case when random() < b.overlap_target then 2 else 1 end, 0
           from turn t
           cross join metric m
           cross join (select overlap_target from batch where id = $1) b
          where t.pool = 'production'
            and t.id = any($2::text[])
            and m.active
            and ($3::text[] is null or m.key = any($3::text[]))
            and (not m.needs_reference or exists (
                  select 1 from dialogue d
                   where d.id = t.dialogue_id and d.reference is not null))
         on conflict do nothing
         returning id`,
        [batchId, turns.map((t) => t.id), metrics],
      );
      console.log(`  batch ${batchId}: ${res.rowCount} items at overlap ${overlap}`);
    }
    await c.query('commit');
  } catch (e) {
    await c.query('rollback');
    throw e;
  } finally {
    await c.end();
  }
}

function chunked<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
