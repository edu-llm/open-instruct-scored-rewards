import { NextResponse } from 'next/server';
import { isBot } from '@/lib/botid';
import { requireQualified } from '@/lib/auth';
import { one, withTx, txOne, txQuery } from '@/lib/db';
import {
  claimItems,
  ensureRaterMetric,
  hydrate,
  insertAssignmentItems,
  isPrimed,
  maybeRetest,
  metricRowTx,
  pickGold,
  pickMetric,
  planRows,
  shouldInjectCheck,
  toMetric,
  type PlannedRow,
} from '@/lib/assign';
import {
  ACTIVE_BATCH_ID,
  ASSIGNMENT_SIZE,
  BREAK_EVERY,
  DAILY_LABEL_CAP,
  PRIMER_SIZE,
} from '@/lib/config';
import { assignmentRequest, parseBody } from '@/lib/validate';
import type { Assignment, AssignmentKind } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  if (await isBot()) return NextResponse.json({ error: 'denied' }, { status: 403 });

  const got = await requireQualified();
  if ('error' in got) return got.error;
  const { rater, session } = got.ok;

  const parsed = await parseBody(req, assignmentRequest);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });
  const size = parsed.data.size ?? ASSIGNMENT_SIZE;

  // Per rater, not per IP: a seminar room on one NAT is exactly the audience this app is for.
  const day = await one<{ n: number }>(
    `select count(*)::int as n from label
      where rater_id = $1 and answered_at > now() - interval '24 hours'`,
    [rater.id],
  );
  if ((day?.n ?? 0) >= DAILY_LABEL_CAP) {
    return NextResponse.json({ error: 'daily_cap' }, { status: 429 });
  }

  const result = await withTx(async (tx) => {
    await txQuery(
      tx,
      `update assignment set completed_at = now()
        where rater_id = $1 and completed_at is null`,
      [rater.id],
    );

    const metricKey = parsed.data.metricKey ?? (await pickMetric(tx, ACTIVE_BATCH_ID, rater.id));
    if (!metricKey) return { error: 'no_work' as const, status: 409 };

    const metric = await metricRowTx(tx, metricKey);
    if (!metric) return { error: 'unknown_metric' as const, status: 404 };

    await ensureRaterMetric(tx, rater.id, metricKey);

    // Two worked answers to the exact question they are about to answer two hundred times. It
    // costs about a minute per rater per metric and it is where most of the protocol effect is.
    let kind: AssignmentKind = (await isPrimed(tx, rater.id, metricKey))
      ? 'production'
      : 'primer';

    let rows: PlannedRow[];
    if (kind === 'primer') {
      const gold = await pickGold(tx, metricKey, rater.id, PRIMER_SIZE, false);
      if (gold.length === 0) {
        // No known answers authored for this metric yet. Mark primed rather than blocking the
        // rater on content that does not exist; the manifest reports the gap.
        await txQuery(
          tx,
          `update rater_metric set primed_at = now() where rater_id = $1 and metric_key = $2`,
          [rater.id, metricKey],
        );
        kind = 'production';
        rows = [];
      } else {
        rows = gold.map((g, i) => ({
          ord: i,
          item_id: null,
          gold_item_id: g.id,
          retest_of: null,
        }));
        await txQuery(
          tx,
          `update rater_metric set primed_at = now() where rater_id = $1 and metric_key = $2`,
          [rater.id, metricKey],
        );
      }
    } else {
      rows = [];
    }

    if (kind === 'production') {
      const claimed = await claimItems(tx, ACTIVE_BATCH_ID, metricKey, rater.id, size);
      if (!claimed.length) return { error: 'no_work' as const, status: 409 };
      const check = shouldInjectCheck()
        ? (await pickGold(tx, metricKey, rater.id, 1, false))[0]
        : undefined;
      rows = planRows(claimed, {
        goldItemId: check?.id ?? null,
        retest: await maybeRetest(tx, rater.id, metricKey),
      });
    }

    const created = await txOne<{ id: string; expires_at: string }>(
      tx,
      `insert into assignment (rater_id, session_id, batch_id, metric_key, kind, n_items)
       values ($1, $2, $3, $4, $5, $6) returning id, expires_at`,
      [rater.id, session.id, ACTIVE_BATCH_ID, metricKey, kind, rows.length],
    );
    if (!created) return { error: 'insert_failed' as const, status: 500 };

    const inserted = await insertAssignmentItems(tx, created.id, rows);
    const items = await hydrate(tx, inserted, metric);

    const counts = await txOne<{ session_labels: number }>(
      tx,
      `select n_labels as session_labels from rater_session where id = $1`,
      [session.id],
    );

    const assignment: Assignment = {
      id: created.id,
      kind,
      metric: toMetric(metric),
      items,
      expiresAt: new Date(created.expires_at).toISOString(),
      progress: {
        doneInSet: 0,
        doneTotal: rater.n_labels,
        sinceBreak: (counts?.session_labels ?? 0) % BREAK_EVERY,
      },
    };
    return { assignment };
  });

  if ('error' in result) {
    return NextResponse.json({ error: result.error }, { status: result.status });
  }
  return NextResponse.json(result.assignment);
}
