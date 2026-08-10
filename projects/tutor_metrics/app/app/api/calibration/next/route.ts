import { NextResponse } from 'next/server';
import { requireRater } from '@/lib/auth';
import { db, one } from '@/lib/db';
import { ensureQuiz, finishQuiz, scoreQuiz } from '@/lib/calibration';
import { hydrate, toMetric, type InsertedAssignmentItem } from '@/lib/assign';
import { getMetricRow } from '@/lib/metrics';
import type { CalibrationNext } from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function GET() {
  const got = await requireRater();
  if ('error' in got) return got.error;
  const { rater } = got.ok;

  if (rater.status === 'active' || rater.status === 'probation') {
    const done: CalibrationNext = {
      done: true,
      passed: true,
      score: rater.calibration_score === null ? null : Number(rater.calibration_score),
      of: 0,
    };
    return NextResponse.json(done);
  }
  if (rater.status === 'failed_calibration' || rater.status === 'suspended') {
    const done: CalibrationNext = {
      done: true,
      passed: false,
      score: rater.calibration_score === null ? null : Number(rater.calibration_score),
      of: 0,
      reason: rater.status,
    };
    return NextResponse.json(done);
  }

  const quiz = await ensureQuiz(got.ok);
  if ('error' in quiz) {
    const done: CalibrationNext = {
      done: true,
      passed: false,
      score: null,
      of: 0,
      reason: quiz.error,
    };
    return NextResponse.json(done, { status: quiz.error === 'no_quiz_items' ? 503 : 200 });
  }

  const next = await one<InsertedAssignmentItem & { metric_key: string }>(
    `select ai.position, ai.presentation_id, ai.item_id, ai.gold_item_id, g.metric_key
       from assignment_item ai
       join gold_item g on g.id = ai.gold_item_id
      where ai.assignment_id = $1 and ai.answered_at is null
      order by ai.position limit 1`,
    [quiz.assignmentId],
  );

  if (!next) {
    const result = await scoreQuiz(quiz.assignmentId);
    await finishQuiz(got.ok, quiz.assignmentId, result);
    const done: CalibrationNext = {
      done: true,
      passed: result.passed,
      score: result.score,
      of: result.of,
      reason: result.passed
        ? undefined
        : result.saysOnlyWrong >= 2
          ? 'says_only_override'
          : 'below_pass_mark',
    };
    return NextResponse.json(done);
  }

  const metric = await getMetricRow(next.metric_key);
  if (!metric) return NextResponse.json({ error: 'unknown_metric' }, { status: 500 });
  const [item] = await hydrate(db(), [next], metric);
  if (!item) return NextResponse.json({ error: 'broken_item' }, { status: 500 });

  const payload: CalibrationNext = {
    done: false,
    item,
    metric: toMetric(metric),
    index: next.position,
    of: quiz.of,
  };
  return NextResponse.json(payload);
}
