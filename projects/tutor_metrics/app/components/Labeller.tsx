'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { LabelStage, type Draft } from './LabelStage';
import type { Assignment, LabelAccepted, LabelSubmission, Progress } from '@/lib/types';

/** Mirrors BREAK_EVERY in lib/config.ts; kept local so no server config reaches the bundle. */
const BREAK_EVERY = 36;

type Phase =
  | { kind: 'loading' }
  | { kind: 'labelling' }
  | { kind: 'break' }
  | { kind: 'nowork' }
  | { kind: 'blocked'; reason: string }
  | { kind: 'error'; reason: string };

const RETRY_DELAYS = [400, 1200, 3000];

async function postWithRetry(url: string, body: unknown): Promise<Response> {
  let last: Response | null = null;
  for (let attempt = 0; attempt <= RETRY_DELAYS.length; attempt += 1) {
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify(body),
      });
      // A 4xx is the server's considered answer, not a transport failure; retrying cannot help.
      if (res.ok || (res.status >= 400 && res.status < 500)) return res;
      last = res;
    } catch {
      last = null;
    }
    if (attempt < RETRY_DELAYS.length) {
      await new Promise((r) => setTimeout(r, RETRY_DELAYS[attempt]));
    }
  }
  return last ?? new Response(JSON.stringify({ error: 'network' }), { status: 599 });
}

export function Labeller() {
  const [phase, setPhase] = useState<Phase>({ kind: 'loading' });
  const [assignment, setAssignment] = useState<Assignment | null>(null);
  const [cursor, setCursor] = useState(0);
  const [answers, setAnswers] = useState<Record<string, Draft>>({});
  const [progress, setProgress] = useState<Progress>({
    doneInSet: 0,
    doneTotal: 0,
    sinceBreak: 0,
  });
  const order = useRef<string[]>([]);

  const loadNext = useCallback(async (metricKey?: string) => {
    setPhase({ kind: 'loading' });
    const res = await postWithRetry('/api/assignment', metricKey ? { metricKey } : {});
    if (res.status === 401) {
      window.location.href = '/';
      return;
    }
    if (res.status === 403) {
      window.location.href = '/calibration';
      return;
    }
    if (res.status === 409 || res.status === 429) {
      setPhase({ kind: 'nowork' });
      return;
    }
    if (!res.ok) {
      setPhase({ kind: 'error', reason: `The server said ${res.status}.` });
      return;
    }
    const next = (await res.json()) as Assignment;
    order.current = [];
    setAnswers({});
    setAssignment(next);
    setCursor(0);
    setProgress(next.progress);
    setPhase({ kind: 'labelling' });
  }, []);

  useEffect(() => {
    let cancelled = false;
    // Read from location rather than useSearchParams so the page needs no Suspense boundary.
    const wanted = new URLSearchParams(window.location.search).get('metric') ?? undefined;
    (async () => {
      if (wanted) {
        await loadNext(wanted);
        return;
      }
      const res = await fetch('/api/assignment/current');
      if (cancelled) return;
      if (res.status === 401) {
        window.location.href = '/';
        return;
      }
      if (res.ok) {
        const open = (await res.json()) as Assignment | null;
        if (open && open.items.length) {
          setAssignment(open);
          setCursor(0);
          setProgress(open.progress);
          setPhase({ kind: 'labelling' });
          return;
        }
      }
      await loadNext();
    })();
    return () => {
      cancelled = true;
    };
  }, [loadNext]);

  const submit = useCallback(
    (draft: Draft, clientDwellMs: number, revisions: number) => {
      if (!assignment) return;
      const item = assignment.items[cursor];
      if (!item) return;

      const body: LabelSubmission = {
        presentationId: item.presentationId,
        value: draft.value,
        naReason: draft.naReason,
        span: draft.span,
        flag: draft.flag,
        clientDwellMs,
        revisions,
      };

      // Recorded and advanced immediately; the POST catches up. Silent loss of a label is the
      // worst failure here, so a POST that will not land stops the session rather than dropping it.
      setAnswers((a) => ({ ...a, [item.presentationId]: draft }));
      order.current = [...order.current, item.presentationId];
      const isLast = cursor >= assignment.items.length - 1;
      if (!isLast) setCursor((c) => c + 1);

      void (async () => {
        const res = await postWithRetry('/api/label', body);
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}) as { error?: string });
          setPhase({
            kind: 'blocked',
            reason:
              `An answer did not save (${(detail as { error?: string }).error ?? res.status}). ` +
              'Nothing further will be recorded until you reload — please do that now.',
          });
          return;
        }
        const accepted = (await res.json()) as LabelAccepted;
        setProgress(accepted.progress);
        if (isLast) {
          if (accepted.breakSuggested) setPhase({ kind: 'break' });
          else void loadNext(assignment.metric.key);
        }
      })();
    },
    [assignment, cursor, loadNext],
  );

  const undo = useCallback(async () => {
    const last = order.current[order.current.length - 1];
    if (!last || !assignment) return;
    const res = await fetch('/api/label/undo', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ presentationId: last }),
    });
    if (!res.ok) return;
    order.current = order.current.slice(0, -1);
    setAnswers((a) => {
      const next = { ...a };
      delete next[last];
      return next;
    });
    const index = assignment.items.findIndex((i) => i.presentationId === last);
    if (index >= 0) setCursor(index);
    setProgress((p) => ({
      doneInSet: Math.max(0, p.doneInSet - 1),
      doneTotal: Math.max(0, p.doneTotal - 1),
      sinceBreak: Math.max(0, p.sinceBreak - 1),
    }));
  }, [assignment]);

  if (phase.kind === 'loading') {
    return <Splash title="Loading your set…" />;
  }
  if (phase.kind === 'error') {
    return (
      <Splash title="Something went wrong">
        <p>{phase.reason}</p>
        <button className="primary" onClick={() => window.location.reload()}>
          Reload
        </button>
      </Splash>
    );
  }
  if (phase.kind === 'nowork') {
    return (
      <Splash title="Nothing left to label right now">
        <p>
          Every item in the current batch either has the labels it needs or has already been shown
          to you. Thank you — that is the whole job done from your side.
        </p>
        <Link href="/me">
          <button className="primary">See your progress</button>
        </Link>
      </Splash>
    );
  }
  if (phase.kind === 'break') {
    return (
      <Splash title="Time for a break">
        <p>
          You have done {progress.doneTotal} in this session. Agreement decays with fatigue faster
          than throughput rises, so this is a real suggestion rather than a courtesy.
        </p>
        <div className="row" style={{ marginTop: 18 }}>
          <Link href="/me">
            <button className="primary">Stop here for now</button>
          </Link>
          <button className="quiet" onClick={() => void loadNext(assignment?.metric.key)}>
            Keep going
          </button>
        </div>
      </Splash>
    );
  }

  if (!assignment) return <Splash title="Loading…" />;
  const item = assignment.items[cursor];
  if (!item) return <Splash title="Loading…" />;

  const doneHere = Object.keys(answers).length;
  const untilBreak = Math.max(0, BREAK_EVERY - progress.sinceBreak);

  return (
    <LabelStage
      metric={assignment.metric}
      item={item}
      counter={`${Math.min(doneHere + 1, assignment.items.length)}/${assignment.items.length}`}
      barPct={doneHere / assignment.items.length}
      note={
        assignment.kind === 'primer'
          ? 'warm-up — answers are shown'
          : `break in ${untilBreak}`
      }
      blocked={phase.kind === 'blocked' ? phase.reason : null}
      answered={answers[item.presentationId] ?? null}
      onSubmit={submit}
      onUndo={() => void undo()}
      onBack={() => setCursor((c) => Math.max(0, c - 1))}
      onForward={() =>
        setCursor((c) =>
          answers[assignment.items[c].presentationId]
            ? Math.min(assignment.items.length - 1, c + 1)
            : c,
        )
      }
    />
  );
}

function Splash({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="page">
      <h1>{title}</h1>
      {children}
    </div>
  );
}
