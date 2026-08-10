/**
 * Push db/metrics.json (rendered from metrics.py) into the metric table.
 *
 *   python scripts/dump_metrics.py
 *   npx dotenv -e .env.local -- npx tsx scripts/seed_metrics.ts
 *
 * metrics.py is the source of truth and this table is a cache. The metric_snapshot trigger keeps
 * every wording it has ever held and refuses a change that does not bump the version, because a
 * wording change retroactively reinterprets every label already collected.
 */
import fs from 'node:fs';
import path from 'node:path';
import { client, flag } from './_pg';
import type { MetricSeedFile } from '../lib/metrics';

interface Existing {
  key: string;
  version: number;
  question: string;
  anchors: unknown;
  guidance: string | null;
  source_sha: string;
}

async function main() {
  const seed = JSON.parse(
    fs.readFileSync(path.join(process.cwd(), 'db', 'metrics.json'), 'utf8'),
  ) as MetricSeedFile;

  const c = await client();
  try {
    const existing = new Map(
      (
        await c.query<Existing>(
          'select key, version, question, anchors, guidance, source_sha from metric',
        )
      ).rows.map((r) => [r.key, r]),
    );

    let inserted = 0;
    let bumped = 0;
    let unchanged = 0;

    for (const m of seed.metrics) {
      const prior = existing.get(m.key);
      const anchorsJson = JSON.stringify(m.anchors);
      const wordingChanged =
        prior !== undefined &&
        (prior.question !== m.question ||
          JSON.stringify(prior.anchors) !== anchorsJson ||
          (prior.guidance ?? null) !== (m.guidance ?? null));
      const version = prior ? (wordingChanged ? prior.version + 1 : prior.version) : 1;

      if (prior && !wordingChanged && prior.source_sha === m.sourceSha) {
        unchanged += 1;
        continue;
      }

      await c.query(
        `insert into metric (key, version, grp, question, anchors, guidance, scale, lo, hi,
                             requires_span, span_from_value, needs_reference, needs_dialogue,
                             allow_na, source_sha, active)
         values ($1,$2,$3::metric_group,$4,$5::jsonb,$6,$7::metric_scale,$8,$9,
                 $10,$11,$12,$13,$14,$15,true)
         on conflict (key) do update set
           version = excluded.version, grp = excluded.grp, question = excluded.question,
           anchors = excluded.anchors, guidance = excluded.guidance, scale = excluded.scale,
           lo = excluded.lo, hi = excluded.hi, requires_span = excluded.requires_span,
           span_from_value = excluded.span_from_value,
           needs_reference = excluded.needs_reference, needs_dialogue = excluded.needs_dialogue,
           allow_na = excluded.allow_na, source_sha = excluded.source_sha, active = true`,
        [
          m.key,
          version,
          m.group,
          m.question,
          anchorsJson,
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
      if (prior) bumped += 1;
      else inserted += 1;
      console.log(`  ${prior ? 'update' : 'insert'} ${m.key} v${version}`);
    }

    const orphans = [...existing.keys()].filter(
      (k) => !seed.metrics.some((m) => m.key === k),
    );
    if (orphans.length) {
      if (flag('deactivate-orphans')) {
        await c.query('update metric set active = false where key = any($1::text[])', [orphans]);
        console.log(`  deactivated ${orphans.join(', ')}`);
      } else {
        console.log(
          `\n  ${orphans.length} metric(s) in the database are not in metrics.py: ${orphans.join(', ')}`,
        );
        console.log('  /api/health will fail until they are removed or --deactivate-orphans is passed.');
      }
    }

    console.log(`\n${inserted} inserted, ${bumped} updated, ${unchanged} unchanged`);
  } finally {
    await c.end();
  }
}

main().catch((e) => {
  console.error(e instanceof Error ? e.message : e);
  process.exit(1);
});
