import { NextResponse } from 'next/server';
import { requireAdmin } from '@/lib/auth';
import { withTx, txQuery } from '@/lib/db';
import { wordCount } from '@/lib/sentences';
import { parseBody, turnsUpload } from '@/lib/validate';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 300;

/**
 * Ingest. Every id arriving here was salted on the researcher's laptop and nothing in the payload
 * identifies a model, a style, a temperature or a scenario target - so there is no provenance in
 * the database to leak, rather than provenance that is merely hidden.
 */
export async function POST(req: Request) {
  const denied = requireAdmin(req);
  if (denied) return denied;

  const parsed = await parseBody(req, turnsUpload);
  if ('error' in parsed) return NextResponse.json({ error: parsed.error }, { status: 400 });
  const { dialogues, turns } = parsed.data;

  const result = await withTx(async (tx) => {
    let insertedDialogues = 0;
    for (const chunk of chunks(dialogues, 200)) {
      const rows = await txQuery<{ id: string }>(
        tx,
        `insert into dialogue (id, subject, grade, question, choices, gold, reference)
         select d.id, d.subject, d.grade, d.question, d.choices, d.gold, d.reference
           from jsonb_to_recordset($1::jsonb) as d(
                id text, subject text, grade int, question text,
                choices jsonb, gold text, reference text)
         on conflict (id) do nothing
         returning id`,
        [
          JSON.stringify(
            chunk.map((d) => ({
              id: d.id,
              subject: d.subject ?? null,
              grade: d.grade ?? null,
              question: d.question,
              choices: d.choices ?? null,
              gold: d.gold ?? null,
              reference: d.reference ?? null,
            })),
          ),
        ],
      );
      insertedDialogues += rows.length;
    }

    let insertedTurns = 0;
    for (const chunk of chunks(turns, 200)) {
      const rows = await txQuery<{ id: string }>(
        tx,
        `insert into turn (id, dialogue_id, pool, turn_index, context,
                           student_before, tutor_turn, n_words, stratum)
         select t.id, t.dialogue_id, t.pool::turn_pool, t.turn_index, t.context,
                t.student_before, t.tutor_turn, t.n_words, t.stratum
           from jsonb_to_recordset($1::jsonb) as t(
                id text, dialogue_id text, pool text, turn_index int, context jsonb,
                student_before text, tutor_turn text, n_words int, stratum text)
         on conflict (id) do nothing
         returning id`,
        [
          JSON.stringify(
            chunk.map((t) => ({
              id: t.id,
              dialogue_id: t.dialogueId,
              pool: t.pool,
              turn_index: t.turnIndex,
              context: t.context,
              student_before: t.studentBefore,
              tutor_turn: t.tutorTurn,
              n_words: wordCount(t.tutorTurn),
              stratum: t.stratum ?? null,
            })),
          ),
        ],
      );
      insertedTurns += rows.length;
    }

    return { insertedDialogues, insertedTurns };
  });

  return NextResponse.json({
    inserted: result.insertedTurns,
    dialogues: result.insertedDialogues,
    skipped: turns.length - result.insertedTurns,
  });
}

function chunks<T>(items: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < items.length; i += size) out.push(items.slice(i, i + size));
  return out;
}
