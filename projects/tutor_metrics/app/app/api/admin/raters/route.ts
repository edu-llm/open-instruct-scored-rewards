import { NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { q } from '@/lib/db';
import type { RaterQcRow, RaterStatus } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const SORTS: Record<string, string> = {
  labels: 'r.n_labels desc',
  recent: 'r.last_seen_at desc',
  gold: 'gold_acc asc nulls last',
  kappa: 'qc.kappa_w asc nulls last',
};

export async function GET(req: Request) {
  const denied = requireAdmin(req);
  if (denied) return denied;

  const params = new URL(req.url).searchParams;
  const status = params.get('status');
  const sort = SORTS[params.get('sort') ?? 'labels'] ?? SORTS.labels;

  const rows = await q<{
    id: string;
    handle: string;
    status: RaterStatus;
    cohort: string | null;
    n_labels: number;
    n_gold_seen: number;
    gold_acc: string | null;
    kappa_w: string | null;
    median_dwell_ms: number | null;
    frac_below_floor: string | null;
    mode_share_excess: string | null;
    runs_z: string | null;
    length_rho: string | null;
    last_seen_at: string;
    qc_note: string | null;
  }>(
    `select r.id, r.handle, r.status, r.cohort, r.n_labels, r.n_gold_seen,
            case when r.n_gold_seen > 0
                 then r.n_gold_correct::numeric / r.n_gold_seen end as gold_acc,
            qc.kappa_w, qc.median_dwell_ms, qc.frac_below_floor,
            qc.mode_share_excess, qc.runs_z, qc.length_rho,
            r.last_seen_at, r.qc_note
       from rater r
       left join lateral (
            select * from rater_qc x
             where x.rater_id = r.id and x.metric_key = '*'
             order by x.computed_at desc limit 1) qc on true
      where ($1::text is null or r.status = $1::rater_status)
      order by ${sort}
      limit 500`,
    [status],
  );

  const out: RaterQcRow[] = rows.map((r) => ({
    raterId: r.id,
    handle: r.handle,
    status: r.status,
    cohort: r.cohort,
    nLabels: r.n_labels,
    goldSeen: r.n_gold_seen,
    goldAcc: r.gold_acc === null ? null : Number(r.gold_acc),
    kappaW: r.kappa_w === null ? null : Number(r.kappa_w),
    medianDwellMs: r.median_dwell_ms,
    fracBelowFloor: r.frac_below_floor === null ? null : Number(r.frac_below_floor),
    modeShareExcess: r.mode_share_excess === null ? null : Number(r.mode_share_excess),
    runsZ: r.runs_z === null ? null : Number(r.runs_z),
    lengthRho: r.length_rho === null ? null : Number(r.length_rho),
    lastSeenAt: r.last_seen_at,
    qcNote: r.qc_note,
  }));
  return NextResponse.json(out);
}
