import { NextResponse } from 'next/server';
import { requireRater } from '@/lib/auth';
import { getMetricRow, toMetric } from '@/lib/metrics';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET(_req: Request, ctx: { params: Promise<{ key: string }> }) {
  const got = await requireRater();
  if ('error' in got) return got.error;
  const { key } = await ctx.params;
  const row = await getMetricRow(key);
  if (!row) return NextResponse.json({ error: 'unknown_metric' }, { status: 404 });
  return NextResponse.json(toMetric(row));
}
