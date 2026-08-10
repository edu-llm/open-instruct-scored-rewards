import { NextResponse } from 'next/server';
import { requireQualified } from '@/lib/auth';
import { submitLabel } from '@/lib/label';
import { labelSubmission, parseBody } from '@/lib/validate';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

/**
 * The hot path. No BotID here: a client that cleared the check on /api/assignment already holds a
 * valid session, and gating every label would cost about $100 per 100,000 for no extra protection.
 *
 * Nothing is rejected for being too fast. A 200 ms answer is stored with its 200 ms and handled by
 * the nightly QC pass; rejecting it would teach an abuser the threshold.
 */
export async function POST(req: Request) {
  const got = await requireQualified();
  if ('error' in got) return got.error;

  const parsed = await parseBody(req, labelSubmission);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });

  // 'primer' too: the two worked answers run through the same screen and the same loop, and the
  // rater is already qualified when they see them.
  const result = await submitLabel(got.ok, parsed.data, ['production', 'primer']);
  if (!result.ok) return NextResponse.json({ error: result.error }, { status: result.status });
  return NextResponse.json(result.body);
}
