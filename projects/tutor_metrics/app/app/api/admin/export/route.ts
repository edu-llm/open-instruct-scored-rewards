import { NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { q } from '@/lib/db';
import { allMetrics } from '@/lib/metrics';
import { buildBundle, type ExportRow, type ExportUnitRow } from '@/lib/export';
import { ACTIVE_BATCH_ID, GUIDE_VERSION } from '@/lib/config';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

const STATUS_SETS: Record<string, string[]> = {
  any: ['new', 'calibrating', 'active', 'probation', 'suspended', 'failed_calibration'],
  active: ['active'],
  probation: ['active', 'probation'],
};

/**
 * The bundle extract_hidden.py and fit_head.py consume unchanged. scripts/unpack_export.py writes
 * it to disk; a zip was rejected because it adds an archive dependency in the function and a
 * stream to get wrong.
 */
export async function GET(req: Request) {
  const denied = requireAdmin(req);
  if (denied) return denied;

  const params = new URL(req.url).searchParams;
  const batch = params.get('batch') ?? ACTIVE_BATCH_ID;
  const minStatus = params.get('min_status') ?? 'probation';
  const statuses = STATUS_SETS[minStatus] ?? STATUS_SETS.probation;

  const rows = await q<ExportRow>(
    `select r.handle, l.turn_id, l.metric_key, l.value, l.na_reason,
            l.flag_kind, l.flag_note, l.gold_item_id, l.retest_of,
            l.span_start, l.span_end, l.span_text, l.sentence_index,
            l.answered_at
       from label l
       join rater r on r.id = l.rater_id
       left join item i on i.id = l.item_id
      where r.status = any($1::rater_status[])
        and (l.item_id is null or i.batch_id = $2)
      order by r.handle, l.turn_id, l.metric_key`,
    [statuses, batch],
  );

  const units = await q<ExportUnitRow>(
    `select distinct tp.id, tp.question, tp.choices, tp.gold, tp.reference,
            tp.student_before, tp.tutor_turn, tp.turn_index, tp.subject, tp.grade
       from turn_public tp
       join label l on l.turn_id = tp.id and l.gold_item_id is null
       join rater r on r.id = l.rater_id and r.status = any($1::rater_status[])`,
    [statuses],
  );

  const metrics = new Map((await allMetrics()).map((m) => [m.key, m]));
  const bundle = buildBundle({
    rows,
    units,
    metrics,
    guideVersion: GUIDE_VERSION,
    filters: { batch, min_status: minStatus, statuses },
  });

  return new NextResponse(JSON.stringify(bundle), {
    headers: {
      'content-type': 'application/json',
      'content-disposition': `attachment; filename="export-${batch}-${Date.now()}.json"`,
    },
  });
}
