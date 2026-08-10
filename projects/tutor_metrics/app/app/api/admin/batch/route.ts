import { NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { withTx, txQuery, txOne } from '@/lib/db';
import { batchRequest, parseBody } from '@/lib/validate';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

/**
 * Materialise (turn, metric) work items. The overlap fraction is data, not code: it is set once,
 * here, so raising it later on a shaky metric is an UPDATE rather than a redeploy.
 */
export async function POST(req: Request) {
  const denied = requireAdmin(req);
  if (denied) return denied;

  const parsed = await parseBody(req, batchRequest);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });
  const { batchId, turnIds, overlapTarget, metrics, note, activate } = parsed.data;

  const out = await withTx(async (tx) => {
    await txQuery(
      tx,
      `insert into batch (id, note, overlap_target, active) values ($1, $2, $3, $4)
       on conflict (id) do update set overlap_target = excluded.overlap_target,
                                      note = coalesce(excluded.note, batch.note),
                                      active = excluded.active`,
      [batchId, note ?? null, overlapTarget, activate],
    );

    const inserted = await txQuery<{ id: string }>(
      tx,
      `insert into item (batch_id, turn_id, metric_key, target_labels, priority)
       select $1, t.id, m.key,
              case when random() < b.overlap_target then 2 else 1 end,
              0
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
      [batchId, turnIds, metrics ?? null],
    );

    // Turns with no reference are quietly excluded from needs_reference metrics, so the count
    // is reported rather than discovered as a shortfall three weeks later.
    const shortfall = await txOne<{ n: number }>(
      tx,
      `select count(*)::int as n from turn t
        where t.id = any($1::text[]) and t.pool = 'production'
          and not exists (select 1 from dialogue d
                           where d.id = t.dialogue_id and d.reference is not null)`,
      [turnIds],
    );

    return { items: inserted.length, turnsWithoutReference: shortfall?.n ?? 0 };
  });

  return NextResponse.json(out);
}
