import { NextResponse } from 'next/server';
import { requireRater } from '@/lib/auth';
import { submitLabel } from '@/lib/label';
import { labelSubmission, parseBody } from '@/lib/validate';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/** Quiz and primer answers carry gold_item_id, so they never reach the export. */
export async function POST(req: Request) {
  const got = await requireRater();
  if ('error' in got) return got.error;

  const parsed = await parseBody(req, labelSubmission);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });

  const result = await submitLabel(got.ok, parsed.data, ['quiz', 'primer']);
  if (!result.ok) return NextResponse.json({ error: result.error }, { status: result.status });
  return NextResponse.json(result.body);
}
