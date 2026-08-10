import { NextResponse } from 'next/server';
import { requireQualified } from '@/lib/auth';
import { one, q } from '@/lib/db';
import { parseBody, undoRequest } from '@/lib/validate';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * Undo deletes the row rather than superseding it, so item.n_labels stays equal to the number of
 * real opinions on the item. The delete branch of label_counters() reverses every counter.
 * Bounded to the rater's own open assignment: labels are never deleted by anyone else, ever.
 */
export async function POST(req: Request) {
  const got = await requireQualified();
  if ('error' in got) return got.error;

  const parsed = await parseBody(req, undoRequest);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });

  const row = await one<{ id: number }>(
    `select l.id from label l
       join assignment a on a.id = l.assignment_id
      where l.presentation_id = $1 and l.rater_id = $2
        and a.expires_at > now()`,
    [parsed.data.presentationId, got.ok.rater.id],
  );
  if (!row) return NextResponse.json({ error: 'not_undoable' }, { status: 404 });

  await q(`delete from label where id = $1`, [row.id]);
  return NextResponse.json({ ok: true });
}
