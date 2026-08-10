import { NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { one, q } from '@/lib/db';
import { ACTIVE_BATCH_ID } from '@/lib/config';
import type { AdminStats, MetricProgressRow, MetricQcRow } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(req: Request) {
  const denied = requireAdmin(req);
  if (denied) return denied;

  const batch = new URL(req.url).searchParams.get('batch') ?? ACTIVE_BATCH_ID;

  const raters = await one<{
    total: number;
    active: number;
    probation: number;
    suspended: number;
    failed: number;
  }>(
    `select count(*)::int                                                as total,
            count(*) filter (where status = 'active')::int               as active,
            count(*) filter (where status = 'probation')::int            as probation,
            count(*) filter (where status = 'suspended')::int            as suspended,
            count(*) filter (where status = 'failed_calibration')::int   as failed
       from rater`,
  );

  const labels = await one<{ total: number; last24h: number }>(
    `select count(*)::int                                                          as total,
            count(*) filter (where answered_at > now() - interval '24 hours')::int as last24h
       from label where gold_item_id is null`,
  );

  const progress = await q<{
    batch_id: string;
    metric_key: string;
    n_items: string;
    target: string;
    got: string;
    complete: string;
    pairs_planned: string;
    pairs_done: string;
  }>(`select * from metric_progress where batch_id = $1 order by metric_key`, [batch]);

  const qc = await q<{
    metric_key: string;
    computed_at: string;
    n_labels: number;
    n_pairs: number;
    kappa_w: string | null;
    flag_rate: string | null;
    na_rate: string | null;
    length_rho: string | null;
    value_histogram: Record<string, number>;
  }>(
    `select distinct on (metric_key) *
       from metric_qc order by metric_key, computed_at desc`,
  );

  const progressRows: MetricProgressRow[] = progress.map((p) => ({
    batchId: p.batch_id,
    metricKey: p.metric_key,
    nItems: Number(p.n_items),
    target: Number(p.target),
    got: Number(p.got),
    complete: Number(p.complete),
    pairsPlanned: Number(p.pairs_planned),
    pairsDone: Number(p.pairs_done),
  }));

  const qcRows: MetricQcRow[] = qc.map((m) => ({
    metricKey: m.metric_key,
    computedAt: m.computed_at,
    nLabels: m.n_labels,
    nPairs: m.n_pairs,
    kappaW: m.kappa_w === null ? null : Number(m.kappa_w),
    flagRate: m.flag_rate === null ? null : Number(m.flag_rate),
    naRate: m.na_rate === null ? null : Number(m.na_rate),
    lengthRho: m.length_rho === null ? null : Number(m.length_rho),
    valueHistogram: m.value_histogram ?? {},
  }));

  // The rubric is on trial too, and these are the standing rules from the guide wired in as
  // alerts rather than left to be noticed.
  const alerts: string[] = [];
  for (const m of qcRows) {
    if (m.flagRate !== null && m.flagRate > 0.2) {
      alerts.push(`${m.metricKey}: flag rate ${(m.flagRate * 100).toFixed(0)}% — rewrite or drop it, whatever its kappa`);
    }
    if (m.kappaW !== null && m.nPairs >= 200 && m.kappaW < 0.4) {
      alerts.push(`${m.metricKey}: kappa_w ${m.kappaW.toFixed(2)} over ${m.nPairs} pairs — below the 0.4 decision rule`);
    }
    if (m.lengthRho !== null && Math.abs(m.lengthRho) > 0.6) {
      alerts.push(`${m.metricKey}: length rho ${m.lengthRho.toFixed(2)} — the pool is scoring length`);
    }
    if (m.naRate !== null && m.naRate > 0.2) {
      alerts.push(`${m.metricKey}: not-applicable rate ${(m.naRate * 100).toFixed(0)}% — the corpus is missing the context, not the rater`);
    }
    const total = Object.values(m.valueHistogram).reduce((s, v) => s + v, 0);
    const top = Math.max(0, ...Object.values(m.valueHistogram));
    if (total >= 100 && top / total > 0.9) {
      alerts.push(`${m.metricKey}: ${(100 * top / total).toFixed(0)}% one value — a prevalence problem more labelling will not fix`);
    }
  }

  const stats: AdminStats = {
    batchId: batch,
    raters: raters ?? { total: 0, active: 0, probation: 0, suspended: 0, failed: 0 },
    labels: labels ?? { total: 0, last24h: 0 },
    progress: progressRows,
    metricQc: qcRows,
    alerts,
  };
  return NextResponse.json(stats);
}
