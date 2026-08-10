import { NextResponse } from 'next/server';
import { requireRater } from '@/lib/auth';
import { raterState } from '@/lib/state';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const got = await requireRater();
  if ('error' in got) return got.error;
  return NextResponse.json(await raterState(got.ok.rater));
}
