import { NextResponse } from 'next/server';
import { one } from '@/lib/db';
import { allMetrics, shaDrift, SEED } from '@/lib/metrics';
import { ACTIVE_BATCH_ID, GUIDE_VERSION, UI_VERSION } from '@/lib/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * metrics.py is the source of truth and the metric table is a cache, so drift between them is a
 * failure and not a warning: the repo has already shipped a run that rated six dimensions against
 * a rubric describing five, and an undefined dimension does not fail loudly on its own.
 */
export async function GET() {
  let dbOk = false;
  let drift: string[] = [];
  let nTurns = 0;
  let nItemsLeft = 0;
  try {
    await one('select 1 as ok');
    dbOk = true;
    drift = shaDrift(await allMetrics());
    const counts = await one<{ turns: number; items_left: number }>(
      `select (select count(*) from turn where pool = 'production')::int as turns,
              (select count(*) from item
                where batch_id = $1 and n_labels < target_labels)::int   as items_left`,
      [ACTIVE_BATCH_ID],
    );
    nTurns = counts?.turns ?? 0;
    nItemsLeft = counts?.items_left ?? 0;
  } catch {
    dbOk = false;
  }

  const ok = dbOk && drift.length === 0;
  return NextResponse.json(
    {
      ok,
      db: dbOk,
      metricShaOk: drift.length === 0,
      drift,
      version: { ui: UI_VERSION, guide: GUIDE_VERSION, metricsFile: SEED.sourceFileSha.slice(0, 12) },
      batch: ACTIVE_BATCH_ID,
      turns: nTurns,
      itemsLeft: nItemsLeft,
    },
    { status: ok ? 200 : 503 },
  );
}
