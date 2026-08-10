import { cookies } from 'next/headers';
import { NextResponse } from 'next/server';
import { isBot } from '@/lib/botid';
import { clientMeta, cookieOptions, COOKIE, createSession, currentCaller } from '@/lib/session';
import { raterState } from '@/lib/state';
import { parseBody, sessionStart } from '@/lib/validate';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  // BotID Deep Analysis is billed per call, so it goes where the call count is small and the
  // value is high: ~1 per rater here, ~1 per 12 labels on /api/assignment, none on /api/label.
  if (await isBot()) return NextResponse.json({ error: 'denied' }, { status: 403 });

  const parsed = await parseBody(req, sessionStart);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });

  const existing = await currentCaller();
  if (existing) return NextResponse.json({ rater: await raterState(existing.rater) });

  const meta = await clientMeta();
  const { token, caller } = await createSession({ src: parsed.data.src ?? null, ...meta });
  const jar = await cookies();
  jar.set(COOKIE, token, cookieOptions());
  return NextResponse.json({ rater: await raterState(caller.rater) });
}
