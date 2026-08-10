import { NextResponse } from 'next/server';
import { requireCron } from '@/lib/auth';
import { q } from '@/lib/db';
import { dwellFloorMs } from '@/lib/config';
import {
  median,
  modeShare,
  probationReasons,
  runsZ,
  spearman,
  suspendReasons,
  weightedKappa,
  type RaterMetricStats,
} from '@/lib/qc';
import type { RaterStatus } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

interface LabelRow {
  id: number;
  rater_id: string;
  handle: string;
  status: RaterStatus;
  turn_id: string;
  metric_key: string;
  value: number | null;
  dwell_ms: number;
  n_words: number;
  flag_kind: string | null;
  lo: number;
  hi: number;
  answered_at: string;
}

export async function GET(req: Request) {
  const denied = requireCron(req);
  if (denied) return denied;

  const rows = await q<LabelRow>(
    `select l.id, l.rater_id, r.handle, r.status, l.turn_id, l.metric_key, l.value,
            l.dwell_ms, t.n_words, l.flag_kind, m.lo, m.hi, l.answered_at
       from label l
       join rater r on r.id = l.rater_id
       join turn  t on t.id = l.turn_id
       join metric m on m.key = l.metric_key
      where l.gold_item_id is null and l.retest_of is null
      order by l.rater_id, l.metric_key, l.answered_at`,
  );

  const gold = await q<{ rater_id: string; metric_key: string; seen: number; correct: number }>(
    `select l.rater_id, l.metric_key, count(*)::int as seen,
            count(*) filter (where l.value is not null
                               and abs(l.value - g.gold_value) <= g.tolerance)::int as correct
       from label l join gold_item g on g.id = l.gold_item_id
      group by 1, 2`,
  );

  const byTurnMetric = new Map<string, LabelRow[]>();
  for (const r of rows) {
    const k = `${r.turn_id}\u241f${r.metric_key}`;
    const list = byTurnMetric.get(k) ?? [];
    list.push(r);
    byTurnMetric.set(k, list);
  }

  // Pool statistics first: mode share is useless raw (length_fit was legitimately 85% one value),
  // and length_rho for a rater only means something against the pool's own coupling.
  const poolMode = new Map<string, number | null>();
  const poolRho = new Map<string, number | null>();
  const byMetric = new Map<string, LabelRow[]>();
  for (const r of rows) {
    const list = byMetric.get(r.metric_key) ?? [];
    list.push(r);
    byMetric.set(r.metric_key, list);
  }
  for (const [key, list] of byMetric) {
    const values = list.map((r) => r.value).filter((v): v is number => v !== null);
    poolMode.set(key, modeShare(values).mode);
    poolRho.set(
      key,
      spearman(
        list.filter((r) => r.value !== null).map((r) => r.value as number),
        list.filter((r) => r.value !== null).map((r) => Math.log(Math.max(r.n_words, 1))),
      ),
    );
  }

  const poolModeShare = new Map<string, number>();
  for (const [key, list] of byMetric) {
    const values = list.map((r) => r.value).filter((v): v is number => v !== null);
    poolModeShare.set(key, modeShare(values).share);
  }

  let metricsScored = 0;
  for (const [key, list] of byMetric) {
    const values = list.map((r) => r.value).filter((v): v is number => v !== null);
    const histogram: Record<string, number> = {};
    for (const v of values) histogram[String(v)] = (histogram[String(v)] ?? 0) + 1;

    const a: number[] = [];
    const b: number[] = [];
    for (const [k, group] of byTurnMetric) {
      if (!k.endsWith(`\u241f${key}`)) continue;
      const scored = group.filter((g) => g.value !== null);
      for (let i = 0; i < scored.length; i += 1) {
        for (let j = i + 1; j < scored.length; j += 1) {
          a.push(scored[i].value as number);
          b.push(scored[j].value as number);
        }
      }
    }
    const lo = list[0]?.lo ?? 1;
    const hi = list[0]?.hi ?? 2;

    await q(
      `insert into metric_qc (metric_key, n_labels, n_pairs, kappa_w, flag_rate, na_rate,
                              length_rho, value_histogram)
       values ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)`,
      [
        key,
        list.length,
        a.length,
        a.length ? weightedKappa(a, b, lo, hi) : null,
        list.filter((r) => r.flag_kind).length / Math.max(list.length, 1),
        list.filter((r) => r.value === null).length / Math.max(list.length, 1),
        poolRho.get(key),
        JSON.stringify(histogram),
      ],
    );
    metricsScored += 1;
  }

  const byRater = new Map<string, LabelRow[]>();
  for (const r of rows) {
    const list = byRater.get(r.rater_id) ?? [];
    list.push(r);
    byRater.set(r.rater_id, list);
  }

  let ratersScored = 0;
  const transitions: string[] = [];

  for (const [raterId, list] of byRater) {
    const status = list[0].status;
    const perMetric = new Map<string, LabelRow[]>();
    for (const r of list) {
      const l = perMetric.get(r.metric_key) ?? [];
      l.push(r);
      perMetric.set(r.metric_key, l);
    }

    const scoped: Array<[string, RaterMetricStats]> = [];
    for (const [key, mine] of perMetric) scoped.push([key, statsFor(mine)]);
    scoped.push(['*', statsFor(list)]);

    for (const [metricKey, s] of scoped) {
      await q(
        `insert into rater_qc (rater_id, metric_key, n_labels, n_overlap, kappa_w, gold_acc,
                               median_dwell_ms, frac_below_floor, mode_share_excess, runs_z,
                               length_rho)
         values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
        [
          raterId,
          metricKey,
          s.nLabels,
          s.nOverlap,
          s.kappaW,
          s.goldAcc,
          s.medianDwellMs,
          s.fracBelowFloor,
          s.modeShareExcess,
          s.runsZ,
          s.lengthRho,
        ],
      );
    }
    ratersScored += 1;

    const pooled = scoped.find(([k]) => k === '*')?.[1];
    if (!pooled) continue;
    const next = nextStatus(status, pooled);
    if (next && next.status !== status) {
      await q(`update rater set status = $2, qc_computed_at = now(), qc_note = $3 where id = $1`, [
        raterId,
        next.status,
        next.reason,
      ]);
      await q(
        `insert into rater_status_event (rater_id, from_status, to_status, reason)
         values ($1, $2, $3, $4)`,
        [raterId, status, next.status, next.reason],
      );
      transitions.push(`${list[0].handle}: ${status} -> ${next.status} (${next.reason})`);
    } else {
      await q(`update rater set qc_computed_at = now() where id = $1`, [raterId]);
    }
  }

  // Earned gold: unanimity rather than a majority, because agreement with the majority is not
  // correctness and a rater who is right where three others were wrong should not be scored down.
  const promoted = await q<{ turn_id: string }>(
    `insert into gold_item (turn_id, metric_key, gold_value, tolerance, rationale, role, source)
     select l.turn_id, l.metric_key, min(l.value), 0,
            'earned: ' || count(*) || ' qualified raters agreed exactly', 'control', 'earned'
       from label l
       join rater r on r.id = l.rater_id and r.status in ('active', 'probation')
      where l.gold_item_id is null and l.retest_of is null and l.value is not null
      group by l.turn_id, l.metric_key
     having count(*) >= 3 and min(l.value) = max(l.value)
     on conflict (turn_id, metric_key) do nothing
     returning turn_id`,
  );

  return NextResponse.json({
    ratersScored,
    metricsScored,
    transitions,
    goldPromoted: promoted.length,
  });

  function statsFor(mine: LabelRow[]): RaterMetricStats {
    const values = mine.map((r) => r.value).filter((v): v is number => v !== null);
    const metricKeys = new Set(mine.map((r) => r.metric_key));

    // kappa_w against the rounded mean of the other raters on double-labelled items.
    const a: number[] = [];
    const b: number[] = [];
    let lo = 1;
    let hi = 2;
    for (const r of mine) {
      if (r.value === null) continue;
      const others = (byTurnMetric.get(`${r.turn_id}\u241f${r.metric_key}`) ?? []).filter(
        (o) => o.rater_id !== r.rater_id && o.value !== null,
      );
      if (!others.length) continue;
      const mean = others.reduce((s, o) => s + (o.value as number), 0) / others.length;
      a.push(r.value);
      b.push(Math.round(mean));
      lo = r.lo;
      hi = r.hi;
    }

    const g = gold.filter(
      (x) => x.rater_id === mine[0].rater_id && (metricKeys.size > 1 || metricKeys.has(x.metric_key)),
    );
    const seen = g.reduce((s, x) => s + x.seen, 0);
    const correct = g.reduce((s, x) => s + x.correct, 0);

    const belowFloor = mine.filter((r) => r.dwell_ms < dwellFloorMs(r.n_words)).length;

    // Mode share excess, per metric against the pool's own share for that metric.
    let excess: number | null = null;
    if (metricKeys.size === 1) {
      const key = [...metricKeys][0];
      excess = modeShare(values).share - (poolModeShare.get(key) ?? 0);
    } else {
      const parts: number[] = [];
      for (const key of metricKeys) {
        const v = mine.filter((r) => r.metric_key === key && r.value !== null).map((r) => r.value as number);
        if (v.length >= 10) parts.push(modeShare(v).share - (poolModeShare.get(key) ?? 0));
      }
      excess = parts.length ? Math.max(...parts) : null;
    }

    const singleMetric = metricKeys.size === 1 ? [...metricKeys][0] : null;
    // Stored raw. The rule from the guide - flag at |rho| > 0.6 only when the pool's rho for that
    // metric is under 0.3 - is applied on the dashboard, where both numbers are side by side.
    const rho = spearman(
      mine.filter((r) => r.value !== null).map((r) => r.value as number),
      mine.filter((r) => r.value !== null).map((r) => Math.log(Math.max(r.n_words, 1))),
    );

    return {
      nLabels: mine.length,
      nOverlap: a.length,
      kappaW: a.length >= 4 ? weightedKappa(a, b, lo, hi) : null,
      goldAcc: seen > 0 ? correct / seen : null,
      medianDwellMs: median(mine.map((r) => r.dwell_ms)),
      fracBelowFloor: mine.length ? belowFloor / mine.length : null,
      modeShareExcess: excess,
      runsZ: runsZ(values, singleMetric ? (poolMode.get(singleMetric) ?? null) : modeShare(values).mode),
      lengthRho: rho,
    };
  }
}

function nextStatus(
  status: RaterStatus,
  s: RaterMetricStats,
): { status: RaterStatus; reason: string } | null {
  const bad = probationReasons(s);
  const worst = suspendReasons(s);

  if (status === 'active' && bad.length) {
    return { status: 'probation', reason: bad.join('; ') };
  }
  if (status === 'probation') {
    if (worst.length) return { status: 'suspended', reason: worst.join('; ') };
    if (!bad.length) return { status: 'active', reason: 'recovered: all screens clear' };
  }
  return null;
}
