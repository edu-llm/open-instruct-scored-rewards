import { NextResponse } from 'next/server';
import { requireCron } from '@/lib/auth';
import { q } from '@/lib/db';
import { allMetrics } from '@/lib/metrics';
import { buildBundle, type ExportRow, type ExportUnitRow } from '@/lib/export';
import { ACTIVE_BATCH_ID, GUIDE_VERSION } from '@/lib/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

/**
 * A nightly export snapshot. Labels are the expensive part of this project, so a copy that does
 * not depend on the database staying healthy is worth one cron slot.
 *
 * Vercel Blob is used when BLOB_READ_WRITE_TOKEN is present and skipped otherwise, so the app
 * deploys and runs without a blob store configured.
 */
export async function GET(req: Request) {
  const denied = requireCron(req);
  if (denied) return denied;

  const rows = await q<ExportRow>(
    `select r.handle, l.turn_id, l.metric_key, l.value, l.na_reason,
            l.flag_kind, l.flag_note, l.gold_item_id, l.retest_of,
            l.span_start, l.span_end, l.span_text, l.sentence_index, l.answered_at
       from label l join rater r on r.id = l.rater_id
      order by r.handle, l.turn_id, l.metric_key`,
  );
  const units = await q<ExportUnitRow>(
    `select distinct tp.id, tp.question, tp.choices, tp.gold, tp.reference,
            tp.student_before, tp.tutor_turn, tp.turn_index, tp.subject, tp.grade
       from turn_public tp join label l on l.turn_id = tp.id and l.gold_item_id is null`,
  );
  const metrics = new Map((await allMetrics()).map((m) => [m.key, m]));
  const bundle = buildBundle({
    rows,
    units,
    metrics,
    guideVersion: GUIDE_VERSION,
    filters: { batch: ACTIVE_BATCH_ID, min_status: 'any', snapshot: true },
  });

  const body = JSON.stringify(bundle);
  if (!process.env.BLOB_READ_WRITE_TOKEN) {
    return NextResponse.json({ blobUrl: null, bytes: body.length, stored: false });
  }

  const name = `snapshots/export-${new Date().toISOString().slice(0, 10)}.json`;
  const res = await fetch(`https://blob.vercel-storage.com/${name}`, {
    method: 'PUT',
    headers: {
      authorization: `Bearer ${process.env.BLOB_READ_WRITE_TOKEN}`,
      'x-api-version': '7',
      'x-content-type': 'application/json',
      'x-add-random-suffix': '0',
      'x-cache-control-max-age': '0',
    },
    body,
  });
  if (!res.ok) {
    return NextResponse.json(
      { error: 'blob_upload_failed', status: res.status, bytes: body.length },
      { status: 502 },
    );
  }
  const blob = (await res.json()) as { url?: string };
  return NextResponse.json({ blobUrl: blob.url ?? null, bytes: body.length, stored: true });
}
