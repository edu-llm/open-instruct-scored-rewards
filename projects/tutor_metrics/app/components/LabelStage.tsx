'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { ReadingColumn } from './ReadingColumn';
import { AnchorList, NaPrompt, RungLadder, SpanPrompt, NA_REASONS } from './AnswerPanel';
import { FlagOverlay, HelpOverlay } from './Overlays';
import { ThemeToggle } from './ThemeToggle';
import type {
  AssignmentItemPayload,
  CalibrationFeedback,
  FlagKind,
  Metric,
  NaReason,
  SpanSubmission,
} from '@/lib/types';

export interface Draft {
  value: number | null;
  naReason?: NaReason;
  span?: SpanSubmission;
  flag?: { kind: FlagKind; note?: string };
}

interface Props {
  metric: Metric;
  item: AssignmentItemPayload;
  counter: string;
  barPct: number;
  note?: string | null;
  busy?: boolean;
  blocked?: string | null;
  feedback?: CalibrationFeedback | null;
  answered?: Draft | null;
  onSubmit: (draft: Draft, clientDwellMs: number, revisions: number) => void;
  onContinue?: () => void;
  onUndo?: () => void;
  onBack?: () => void;
  onForward?: () => void;
}

type Phase = 'score' | 'span' | 'na';

export function LabelStage({
  metric,
  item,
  counter,
  barPct,
  note,
  busy = false,
  blocked = null,
  feedback = null,
  answered = null,
  onSubmit,
  onContinue,
  onUndo,
  onBack,
  onForward,
}: Props) {
  const [phase, setPhase] = useState<Phase>('score');
  const [value, setValue] = useState<number | null>(null);
  const [flag, setFlag] = useState<{ kind: FlagKind; note?: string } | undefined>();
  const [overlay, setOverlay] = useState<'help' | 'flag' | null>(null);
  const [dragged, setDragged] = useState<SpanSubmission | null>(null);
  const [fading, setFading] = useState(false);

  const shownAt = useRef(Date.now());
  const revisions = useRef(0);

  // Advancing to the next item is a state change, not a network round-trip, so the only motion is
  // a 90 ms fade on the turn text. Nothing slides and nothing reflows.
  useEffect(() => {
    setPhase('score');
    setValue(null);
    setFlag(undefined);
    setDragged(null);
    setOverlay(null);
    revisions.current = 0;
    shownAt.current = Date.now();
    setFading(true);
    const t = setTimeout(() => setFading(false), 20);
    return () => clearTimeout(t);
  }, [item.presentationId]);

  const readOnly = answered !== null || feedback !== null;

  const commit = useCallback(
    (draft: Draft) => {
      if (busy || blocked || readOnly) return;
      onSubmit(draft, Date.now() - shownAt.current, revisions.current);
    },
    [busy, blocked, readOnly, onSubmit],
  );

  const spanDemanded = useCallback(
    (v: number) =>
      metric.requiresSpan && (metric.spanFromValue === null || v >= metric.spanFromValue),
    [metric.requiresSpan, metric.spanFromValue],
  );

  const pickValue = useCallback(
    (v: number) => {
      if (readOnly || busy || blocked) return;
      if (value !== null && value !== v) revisions.current += 1;
      setValue(v);
      if (!spanDemanded(v)) {
        commit({ value: v, flag });
        return;
      }
      if (dragged) {
        commit({ value: v, span: dragged, flag });
        return;
      }
      const sentences = item.turn.sentences;
      if (sentences.length === 1) {
        const s = sentences[0];
        commit({
          value: v,
          span: { start: s.start, end: s.end, text: s.text, sentenceIndex: 0 },
          flag,
        });
        return;
      }
      setPhase('span');
    },
    [readOnly, busy, blocked, value, spanDemanded, dragged, commit, flag, item.turn.sentences],
  );

  const pickSentence = useCallback(
    (index: number) => {
      const s = item.turn.sentences[index];
      if (!s || value === null) return;
      commit({
        value,
        span: { start: s.start, end: s.end, text: s.text, sentenceIndex: index },
        flag,
      });
    },
    [item.turn.sentences, value, commit, flag],
  );

  const dragSpan = useCallback(
    (span: { start: number; end: number; text: string }) => {
      const full: SpanSubmission = { ...span, sentenceIndex: null };
      if (phase === 'span' && value !== null) {
        commit({ value, span: full, flag });
        return;
      }
      setDragged(full);
    },
    [phase, value, commit, flag],
  );

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
        if (e.key === 'Escape') target.blur();
        return;
      }
      // Deliberately inert, so a stray return cannot submit.
      if (e.key === 'Enter') {
        e.preventDefault();
        return;
      }
      if (overlay === 'flag') return;
      if (overlay === 'help') {
        if (e.key === 'Escape' || e.key === '?') setOverlay(null);
        return;
      }
      if (e.key === '?') {
        e.preventDefault();
        setOverlay('help');
        return;
      }
      if (feedback) {
        if (e.key === ' ') {
          e.preventDefault();
          onContinue?.();
        }
        return;
      }
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        onBack?.();
        return;
      }
      if (e.key === 'ArrowRight') {
        e.preventDefault();
        onForward?.();
        return;
      }
      if (e.key === 'u') {
        e.preventDefault();
        onUndo?.();
        return;
      }
      if (readOnly || busy || blocked) return;

      if (e.key === 'Escape') {
        if (phase !== 'score') {
          setPhase('score');
          setValue(null);
        }
        return;
      }
      if (e.key === 'f') {
        e.preventDefault();
        setOverlay('flag');
        return;
      }
      if (e.key === 'n' && metric.allowNa) {
        e.preventDefault();
        setPhase(phase === 'na' ? 'score' : 'na');
        return;
      }

      const n = Number(e.key);
      if (!Number.isInteger(n) || e.key.length !== 1) return;
      if (phase === 'span') {
        if (n >= 1 && n <= Math.min(9, item.turn.sentences.length)) {
          e.preventDefault();
          pickSentence(n - 1);
        }
        return;
      }
      if (phase === 'na') {
        if (n >= 1 && n <= NA_REASONS.length) {
          e.preventDefault();
          commit({ value: null, naReason: NA_REASONS[n - 1].key, flag });
        }
        return;
      }
      // Out-of-range keys are ignored, not clamped.
      if (n >= metric.lo && n <= metric.hi) {
        e.preventDefault();
        pickValue(n);
      }
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [
    overlay,
    phase,
    metric.allowNa,
    metric.lo,
    metric.hi,
    item.turn.sentences.length,
    pickSentence,
    pickValue,
    commit,
    flag,
    feedback,
    readOnly,
    busy,
    blocked,
    onBack,
    onForward,
    onUndo,
    onContinue,
  ]);

  const shownValue = answered ? answered.value : value;
  // Named, ordered and more than two of them: the ladder. A categorical with names - icap_class -
  // is deliberately not drawn as a ladder, because its classes are not ordered.
  const isLadder =
    metric.scale === 'ordinal' && metric.hi - metric.lo >= 2 && metric.anchors.every((a) => a.name);

  return (
    <div className="shell">
      <header className="topbar">
        <span className="key">{metric.key}</span>
        <div className="bar">
          <i style={{ width: `${Math.round(barPct * 100)}%` }} />
        </div>
        <span className="meta">{counter}</span>
        {note ? <span className="meta">{note}</span> : null}
        <span className="spacer" />
        {flag ? <span className="chip warn">flagged: {flag.kind}</span> : null}
        <Link href="/me" className="meta">
          progress
        </Link>
        <ThemeToggle />
        <button className="quiet" onClick={() => setOverlay('help')} aria-label="Anchors and rules">
          <kbd>?</kbd>
        </button>
      </header>

      <div className="stage">
        <ReadingColumn
          turn={item.turn}
          metric={metric}
          picking={phase === 'span'}
          marked={dragged}
          fading={fading}
          onPickSentence={pickSentence}
          onDragSpan={dragSpan}
        />

        <div className="answer">
          <p className="question">{metric.question}</p>

          {isLadder ? (
            <RungLadder
              metric={metric}
              selected={shownValue}
              disabled={readOnly || busy || Boolean(blocked)}
              onPick={pickValue}
            />
          ) : (
            <AnchorList
              metric={metric}
              selected={shownValue}
              disabled={readOnly || busy || Boolean(blocked)}
              onPick={pickValue}
            />
          )}

          {dragged && phase !== 'span' ? (
            <div className="row" style={{ marginTop: 10 }}>
              <span className="chip good">evidence selected</span>
              <button className="quiet" onClick={() => setDragged(null)}>
                clear
              </button>
            </div>
          ) : null}

          {phase === 'span' && !readOnly ? (
            <SpanPrompt
              count={Math.min(9, item.turn.sentences.length)}
              chosen={null}
              onPick={pickSentence}
              onCancel={() => {
                setPhase('score');
                setValue(null);
              }}
            />
          ) : null}

          {phase === 'na' && !readOnly ? (
            <NaPrompt
              onPick={(reason) => commit({ value: null, naReason: reason, flag })}
              onCancel={() => setPhase('score')}
            />
          ) : null}

          {answered ? (
            <div className="notice" style={{ marginTop: 12 }}>
              Already answered
              {answered.naReason ? ` — not applicable (${answered.naReason})` : ''}. Press{' '}
              <kbd>u</kbd> to undo it, or <kbd>→</kbd> to move on.
            </div>
          ) : null}

          {feedback ? (
            <div className={`feedback ${feedback.correct ? 'right' : 'wrong'}`}>
              <div className="verdict">
                <span aria-hidden>{feedback.correct ? '✓' : '✕'}</span>
                {feedback.correct ? 'Correct' : `Not this one — the answer is ${feedback.goldValue}`}
                <span className="spacer" />
                <span className="chip">
                  {feedback.running.correct}/{feedback.running.answered} so far
                </span>
              </div>
              <p>{feedback.rationale}</p>
              <div className="row" style={{ marginTop: 12 }}>
                <button className="primary" onClick={onContinue}>
                  Next <kbd style={{ marginLeft: 6 }}>space</kbd>
                </button>
              </div>
            </div>
          ) : null}

          <div className="hints">
            <span>
              <kbd>{metric.lo}</kbd>–<kbd>{metric.hi}</kbd> answer
            </span>
            {metric.allowNa ? (
              <span>
                <kbd>n</kbd> n/a
              </span>
            ) : null}
            <span>
              <kbd>f</kbd> flag
            </span>
            <span>
              <kbd>u</kbd> undo
            </span>
            <span>
              <kbd>?</kbd> anchors
            </span>
          </div>
        </div>
      </div>

      {overlay === 'help' ? (
        <HelpOverlay metric={metric} onClose={() => setOverlay(null)} />
      ) : null}
      {overlay === 'flag' ? (
        <FlagOverlay
          initial={flag}
          onSave={(f) => {
            setFlag(f ?? undefined);
            setOverlay(null);
          }}
          onClose={() => setOverlay(null)}
        />
      ) : null}

      {blocked ? <div className="banner">{blocked}</div> : null}
    </div>
  );
}
