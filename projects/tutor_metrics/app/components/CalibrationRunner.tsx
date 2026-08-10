'use client';

import { useCallback, useEffect, useState } from 'react';
import Link from 'next/link';
import { LabelStage, type Draft } from './LabelStage';
import { RULES } from './Overlays';
import type {
  AssignmentItemPayload,
  CalibrationFeedback,
  CalibrationNext,
  LabelAccepted,
  Metric,
} from '@/lib/types';

interface Live {
  item: AssignmentItemPayload;
  metric: Metric;
  index: number;
  of: number;
}

/**
 * Fifteen known-answer items with feedback after every one. This is the calibration session from
 * the guide rendered as software: the reliability gain comes from resolving disagreements into
 * stated rules, not from excluding people, so nothing here is hidden from the rater.
 */
export function CalibrationRunner() {
  const [live, setLive] = useState<Live | null>(null);
  const [feedback, setFeedback] = useState<CalibrationFeedback | null>(null);
  const [answered, setAnswered] = useState<Draft | null>(null);
  const [done, setDone] = useState<Extract<CalibrationNext, { done: true }> | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setFeedback(null);
    setAnswered(null);
    const res = await fetch('/api/calibration/next');
    if (res.status === 401) {
      window.location.href = '/';
      return;
    }
    const body = (await res.json()) as CalibrationNext | { error: string };
    if ('error' in body) {
      setError(body.error);
      return;
    }
    if (body.done) {
      setDone(body);
      setLive(null);
      return;
    }
    setLive({ item: body.item, metric: body.metric, index: body.index, of: body.of });
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const submit = useCallback(
    async (draft: Draft, clientDwellMs: number, revisions: number) => {
      if (!live) return;
      setAnswered(draft);
      const res = await fetch('/api/calibration/answer', {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          presentationId: live.item.presentationId,
          value: draft.value,
          naReason: draft.naReason,
          span: draft.span,
          flag: draft.flag,
          clientDwellMs,
          revisions,
        }),
      });
      if (!res.ok) {
        const detail = (await res.json().catch(() => ({}))) as { error?: string };
        setError(detail.error ?? `server said ${res.status}`);
        return;
      }
      const accepted = (await res.json()) as LabelAccepted;
      setFeedback(accepted.feedback ?? null);
      if (!accepted.feedback) void load();
    },
    [live, load],
  );

  if (error === 'no_quiz_items') {
    return (
      <div className="page">
        <h1>The calibration set is not loaded yet</h1>
        <p>
          No known-answer items have been authored for the active metrics, so there is nothing to
          calibrate against. This is a setup step on our side, not something you did.
        </p>
        <p className="mono" style={{ fontSize: 13 }}>
          researcher: POST the quiz items to /api/admin/gold with useInQuiz true
        </p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="page">
        <h1>Something went wrong</h1>
        <p className="mono">{error}</p>
        <button className="primary" onClick={() => window.location.reload()}>
          Reload
        </button>
      </div>
    );
  }

  if (done) return <Verdict done={done} />;
  if (!live) return <div className="page"><h1>Loading the calibration set…</h1></div>;

  return (
    <LabelStage
      metric={live.metric}
      item={live.item}
      counter={`${live.index + 1}/${live.of}`}
      barPct={live.index / Math.max(live.of, 1)}
      note="calibration — the answer is shown after each one"
      feedback={feedback}
      answered={feedback ? null : answered}
      onSubmit={(d, dwell, rev) => void submit(d, dwell, rev)}
      onContinue={() => void load()}
    />
  );
}

function Verdict({ done }: { done: Extract<CalibrationNext, { done: true }> }) {
  if (done.passed) {
    return (
      <div className="page">
        <h1>You are calibrated</h1>
        <p className="lead">
          {done.score !== null ? `${done.score} of ${done.of}.` : ''} Your labels count from here.
        </p>
        <div className="notice good" style={{ marginTop: 20 }}>
          One metric at a time, twelve turns at a time. The anchors stay on screen; press{' '}
          <kbd>?</kbd> whenever a boundary case comes up.
        </div>
        <div className="row" style={{ marginTop: 24 }}>
          <Link href="/label">
            <button className="primary">Start labelling</button>
          </Link>
        </div>
      </div>
    );
  }

  const noAttempts = done.reason === 'no_attempts_left' || done.reason === 'failed_calibration';
  return (
    <div className="page">
      <h1>{noAttempts ? 'Thank you for trying' : 'Not quite — one more attempt'}</h1>
      <p className="lead">
        {done.score !== null ? `You scored ${done.score} of ${done.of}. ` : ''}
        {done.reason === 'says_only_override'
          ? 'Several turns that talked about a teaching move without making it were marked as making it. That single confusion produces a confident, plausible, wrong column, so it is scored strictly.'
          : 'The pass mark is 70%.'}
      </p>
      <h2>The four rules that move agreement most</h2>
      <ol>
        {RULES.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ol>
      {noAttempts ? (
        <p>
          Your answers are kept and reported, and nothing is discarded silently. We are grateful for
          the time — the calibration gate exists because a pool of labels that disagree with each
          other is worth less than a smaller pool that does not.
        </p>
      ) : (
        <div className="row" style={{ marginTop: 24 }}>
          <button className="primary" onClick={() => window.location.reload()}>
            Try the second set
          </button>
        </div>
      )}
    </div>
  );
}
