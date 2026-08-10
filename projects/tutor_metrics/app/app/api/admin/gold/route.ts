import { NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { q } from '@/lib/db';
import { goldUpload, parseBody } from '@/lib/validate';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

export async function POST(req: Request) {
  const denied = requireAdmin(req);
  if (denied) return denied;

  const parsed = await parseBody(req, goldUpload);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });

  // assert_pool() rejects an authored answer on a production turn: a pool that labels its own
  // answers measures the labeller's belief about the answer.
  const rows = await q<{ id: number }>(
    `insert into gold_item (turn_id, metric_key, gold_value, tolerance, rationale, role,
                            use_in_quiz, quiz_position, source)
     select g.turn_id, g.metric_key, g.gold_value, g.tolerance, g.rationale, g.role,
            g.use_in_quiz, g.quiz_position, 'authored'
       from jsonb_to_recordset($1::jsonb) as g(
            turn_id text, metric_key text, gold_value smallint, tolerance smallint,
            rationale text, role text, use_in_quiz boolean, quiz_position int)
     on conflict (turn_id, metric_key) do nothing
     returning id`,
    [
      JSON.stringify(
        parsed.data.goldItems.map((g) => ({
          turn_id: g.turnId,
          metric_key: g.metricKey,
          gold_value: g.goldValue,
          tolerance: g.tolerance,
          rationale: g.rationale,
          role: g.role,
          use_in_quiz: g.useInQuiz,
          quiz_position: g.quizPosition ?? null,
        })),
      ),
    ],
  );

  return NextResponse.json({
    inserted: rows.length,
    skipped: parsed.data.goldItems.length - rows.length,
  });
}
