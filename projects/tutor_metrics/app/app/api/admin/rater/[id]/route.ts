import { NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { txQuery, txOne, withTx } from '@/lib/db';
import { parseBody, raterPatch } from '@/lib/validate';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * Status is never applied retroactively to the data. Labels are not deleted here or anywhere -
 * the export filters by status at export time, so the effect of excluding a cohort can be
 * measured rather than assumed.
 */
export async function PATCH(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const denied = requireAdmin(req);
  if (denied) return denied;

  const { id } = await ctx.params;
  const parsed = await parseBody(req, raterPatch);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });

  const ok = await withTx(async (tx) => {
    const before = await txOne<{ status: string }>(
      tx,
      `select status from rater where id = $1`,
      [id],
    );
    if (!before) return false;
    await txQuery(
      tx,
      `update rater set status = $2::rater_status,
                        qualified_at = case when $2::text = 'active' and qualified_at is null
                                            then now() else qualified_at end
        where id = $1`,
      [id, parsed.data.status],
    );
    await txQuery(
      tx,
      `insert into rater_status_event (rater_id, from_status, to_status, reason)
       values ($1, $2, $3, $4)`,
      [id, before.status, parsed.data.status, parsed.data.reason],
    );
    return true;
  });

  if (!ok) return NextResponse.json({ error: 'unknown_rater' }, { status: 404 });
  return NextResponse.json({ ok: true });
}
