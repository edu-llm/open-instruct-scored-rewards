import { NextResponse } from 'next/server';
import { requireRater } from '@/lib/auth';
import { db, one, q } from '@/lib/db';
import { hydrate, toMetric, type InsertedAssignmentItem } from '@/lib/assign';
import { getMetricRow } from '@/lib/metrics';
import { BREAK_EVERY } from '@/lib/config';
import type { Assignment, AssignmentKind } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** Resuming after a reload. Only unanswered positions come back, so nothing is asked twice. */
export async function GET() {
  const got = await requireRater();
  if ('error' in got) return got.error;
  const { rater, session } = got.ok;

  const open = await one<{
    id: string;
    kind: AssignmentKind;
    metric_key: string | null;
    expires_at: string;
    n_done: number;
  }>(
    `select id, kind, metric_key, expires_at, n_done
       from assignment
      where rater_id = $1 and completed_at is null and expires_at > now()
      order by created_at desc limit 1`,
    [rater.id],
  );
  if (!open || !open.metric_key) return NextResponse.json(null);

  const metric = await getMetricRow(open.metric_key);
  if (!metric) return NextResponse.json(null);

  const pending = await q<InsertedAssignmentItem>(
    `select position, presentation_id, item_id, gold_item_id
       from assignment_item
      where assignment_id = $1 and answered_at is null
      order by position`,
    [open.id],
  );
  if (!pending.length) return NextResponse.json(null);

  const items = await hydrate(db(), pending, metric);
  const counts = await one<{ session_labels: number }>(
    `select n_labels as session_labels from rater_session where id = $1`,
    [session.id],
  );

  const assignment: Assignment = {
    id: open.id,
    kind: open.kind,
    metric: toMetric(metric),
    items,
    expiresAt: new Date(open.expires_at).toISOString(),
    progress: {
      doneInSet: open.n_done,
      doneTotal: rater.n_labels,
      sinceBreak: (counts?.session_labels ?? 0) % BREAK_EVERY,
    },
  };
  return NextResponse.json(assignment);
}
