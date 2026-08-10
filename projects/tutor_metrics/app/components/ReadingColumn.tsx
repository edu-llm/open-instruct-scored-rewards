'use client';

import { Fragment, useMemo } from 'react';
import type { Metric, TurnPublic } from '@/lib/types';

function norm(s: string | null | undefined): string {
  return (s ?? '').trim().replace(/\s+/g, ' ').toLowerCase();
}

function peek(text: string, n = 90): string {
  const flat = text.replace(/\s+/g, ' ').trim();
  return flat.length > n ? `${flat.slice(0, n)}…` : flat;
}

interface Props {
  turn: TurnPublic;
  metric: Metric;
  picking: boolean;
  /** Character range of a dragged selection being carried into the next answer. */
  marked: { start: number; end: number } | null;
  fading: boolean;
  onPickSentence: (index: number) => void;
  onDragSpan: (span: { start: number; end: number; text: string }) => void;
}

/**
 * Which panels appear is driven by the metric row, not hard-coded. The student's previous message
 * is always shown and never collapsible - several metrics are defined against it - and the tutor
 * turn is the only thing on the page read word by word.
 */
export function ReadingColumn({
  turn,
  metric,
  picking,
  marked,
  fading,
  onPickSentence,
  onDragSpan,
}: Props) {
  const goldInChoices = useMemo(
    () => (turn.choices ?? []).some((c) => turn.gold != null && norm(c) === norm(turn.gold)),
    [turn.choices, turn.gold],
  );

  const pieces = useMemo(() => {
    const out: Array<{ gap: string; index: number; start: number; text: string }> = [];
    let cursor = 0;
    for (const s of turn.sentences) {
      out.push({
        gap: turn.tutorTurn.slice(cursor, s.start),
        index: s.index,
        start: s.start,
        text: turn.tutorTurn.slice(s.start, s.end),
      });
      cursor = s.end;
    }
    return { parts: out, tail: turn.tutorTurn.slice(cursor) };
  }, [turn.sentences, turn.tutorTurn]);

  const numbered = turn.sentences.length > 1;

  function handleMouseUp() {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) return;
    const from = resolve(sel.anchorNode, sel.anchorOffset);
    const to = resolve(sel.focusNode, sel.focusOffset);
    if (from === null || to === null) return;
    const start = Math.min(from, to);
    const end = Math.max(from, to);
    if (end - start < 2) return;
    // Dragging overrides the sentence choice for the cases where the sentence is the wrong unit.
    onDragSpan({ start, end, text: turn.tutorTurn.slice(start, end) });
    sel.removeAllRanges();
  }

  return (
    <div className="reading">
      <details className="panel">
        <summary>
          <span className="label">The problem</span>
          <span className="peek">{peek(turn.question)}</span>
        </summary>
        <div className="panel-body">
          <span className="ink">{turn.question}</span>
          {turn.choices && turn.choices.length > 0 ? (
            <ul className="choices">
              {turn.choices.map((c, i) => (
                <li
                  key={`${i}-${c}`}
                  className={turn.gold != null && norm(c) === norm(turn.gold) ? 'correct' : ''}
                >
                  {String.fromCharCode(65 + i)}. {c}
                  {turn.gold != null && norm(c) === norm(turn.gold) ? ' — correct answer' : ''}
                </li>
              ))}
            </ul>
          ) : null}
          {turn.gold && !goldInChoices ? (
            <div style={{ marginTop: 10, color: 'var(--good)' }}>Correct answer: {turn.gold}</div>
          ) : null}
        </div>
      </details>

      {metric.needsReference && turn.reference ? (
        <details className="panel" open>
          <summary>
            <span className="label">Worked solution</span>
            <span className="peek">check the turn against this — do not re-derive it</span>
          </summary>
          <div className="panel-body">{turn.reference}</div>
        </details>
      ) : null}

      {metric.needsDialogue && turn.context.length > 0 ? (
        <details className="panel">
          <summary>
            <span className="label">Earlier in the dialogue</span>
            <span className="peek">{turn.context.length} turns before this one</span>
          </summary>
          <div className="panel-body">
            {turn.context.map((c, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <span className="eyebrow" style={{ marginBottom: 2 }}>
                  {c.role}
                </span>
                {c.text}
              </div>
            ))}
          </div>
        </details>
      ) : null}

      <div className="block">
        <div className="eyebrow">Student, just before</div>
        <div className="student">{turn.studentBefore}</div>
      </div>

      <div className="turn-wrap">
        <div className="eyebrow">
          Tutor turn — rate this
          {picking ? <span className="chip warn">choose the part that does it</span> : null}
        </div>
        <div
          className={`turn${picking ? ' picking' : ''}${fading ? ' fading' : ''}`}
          onMouseUp={handleMouseUp}
        >
          {pieces.parts.map((p) => (
            <Fragment key={p.index}>
              {p.gap}
              {numbered ? <sup className="sent-n">{p.index + 1}</sup> : null}
              <span
                className={`sent${
                  marked && p.start < marked.end && p.start + p.text.length > marked.start
                    ? ' chosen'
                    : ''
                }`}
                data-start={p.start}
                onClick={picking ? () => onPickSentence(p.index) : undefined}
              >
                {p.text}
              </span>
            </Fragment>
          ))}
          {pieces.tail}
        </div>
      </div>
    </div>
  );
}

function resolve(node: Node | null, offset: number): number | null {
  if (!node) return null;
  const el = node.nodeType === Node.TEXT_NODE ? node.parentElement : (node as Element);
  const host = el?.closest<HTMLElement>('[data-start]');
  if (!host) return null;
  return Number(host.dataset.start) + offset;
}
