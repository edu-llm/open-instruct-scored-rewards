'use client';

import { useEffect, useRef, useState } from 'react';
import type { FlagKind, Metric } from '@/lib/types';

export const FLAG_KINDS: Array<{ key: FlagKind; label: string }> = [
  { key: 'rubric_misfit', label: 'The question does not fit this turn' },
  { key: 'says_only', label: 'The turn talks about the thing without doing it' },
  { key: 'factual_error', label: 'The turn states something false' },
  { key: 'broken_item', label: 'The item is broken — truncated, empty or garbled' },
];

/** The four rules, verbatim from the guide, because they are the ones that move agreement. */
export const RULES = [
  'You are judging whether the turn does the thing, not whether it is good teaching.',
  'Answer by pointing at the text. If you have to form an overall impression, flag it.',
  'Check factual claims against the reference solution shown; do not re-derive them.',
  'You are not judging whether a different response would have been better.',
];

export function HelpOverlay({ metric, onClose }: { metric: Metric; onClose: () => void }) {
  return (
    <div className="scrim" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="sheet" onClick={(e) => e.stopPropagation()}>
        <div className="row">
          <h3 className="mono">{metric.key}</h3>
          <span className="spacer" />
          <button className="quiet" onClick={onClose}>
            <kbd>esc</kbd> close
          </button>
        </div>
        <p className="lead" style={{ color: 'var(--ink)', fontSize: 16 }}>
          {metric.question}
        </p>

        <h4>Anchors</h4>
        {metric.anchors.map((a) => (
          <div key={a.value} className="row" style={{ alignItems: 'flex-start', marginBottom: 10 }}>
            <kbd>{a.value}</kbd>
            <div style={{ flex: 1, minWidth: 0 }}>
              {a.name ? (
                <div className="mono" style={{ fontWeight: 700, fontSize: 13.5 }}>
                  {a.name}
                </div>
              ) : null}
              <div style={{ color: 'var(--dim)', fontSize: 14.5 }}>{a.label}</div>
            </div>
          </div>
        ))}

        {metric.guidance ? (
          <>
            <h4>Guidance</h4>
            <p style={{ whiteSpace: 'pre-wrap' }}>{metric.guidance}</p>
          </>
        ) : null}

        <h4>How to answer</h4>
        <ol>
          {RULES.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ol>

        <h4>Keys</h4>
        <div className="hints">
          <span>
            <kbd>{metric.lo}</kbd>–<kbd>{metric.hi}</kbd> score
          </span>
          <span>
            <kbd>1</kbd>–<kbd>9</kbd> sentence, when evidence is asked for
          </span>
          {metric.allowNa ? (
            <span>
              <kbd>n</kbd> not applicable
            </span>
          ) : null}
          <span>
            <kbd>f</kbd> flag
          </span>
          <span>
            <kbd>u</kbd> undo
          </span>
          <span>
            <kbd>←</kbd>
            <kbd>→</kbd> move
          </span>
          <span>
            <kbd>?</kbd> this panel
          </span>
        </div>
      </div>
    </div>
  );
}

interface FlagProps {
  initial?: { kind: FlagKind; note?: string };
  onSave: (flag: { kind: FlagKind; note?: string } | null) => void;
  onClose: () => void;
}

export function FlagOverlay({ initial, onSave, onClose }: FlagProps) {
  const [kind, setKind] = useState<FlagKind | null>(initial?.kind ?? null);
  const [note, setNote] = useState(initial?.note ?? '');
  const box = useRef<HTMLInputElement>(null);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        if (document.activeElement === box.current) {
          box.current?.blur();
          return;
        }
        onClose();
        return;
      }
      if (document.activeElement === box.current) return;
      const n = Number(e.key);
      if (n >= 1 && n <= FLAG_KINDS.length) {
        e.preventDefault();
        setKind(FLAG_KINDS[n - 1].key);
      }
    }
    document.addEventListener('keydown', onKey, true);
    return () => document.removeEventListener('keydown', onKey, true);
  }, [onClose]);

  return (
    <div className="scrim" role="dialog" aria-modal="true" onClick={onClose}>
      <div className="sheet" style={{ maxWidth: 560 }} onClick={(e) => e.stopPropagation()}>
        <h3>Flag this item</h3>
        <p>
          A high flag rate on a metric means the question does not fit the turns, which is a low
          agreement arriving more politely. Flagging is more useful than guessing.
        </p>
        <div className="pick-row" style={{ flexDirection: 'column', marginTop: 14 }}>
          {FLAG_KINDS.map((f, i) => (
            <button
              key={f.key}
              type="button"
              className={`pick${kind === f.key ? ' sel' : ''}`}
              onClick={() => setKind(f.key)}
            >
              <kbd>{i + 1}</kbd>
              <span>{f.label}</span>
            </button>
          ))}
        </div>
        <div style={{ marginTop: 14 }}>
          <input
            ref={box}
            type="text"
            placeholder="Optional note — one line is plenty"
            value={note}
            onChange={(e) => setNote(e.target.value)}
          />
        </div>
        <div className="row" style={{ marginTop: 16 }}>
          {initial ? (
            <button onClick={() => onSave(null)}>Remove flag</button>
          ) : (
            <button className="quiet" onClick={onClose}>
              Cancel
            </button>
          )}
          <span className="spacer" />
          <button
            className="primary"
            disabled={!kind}
            onClick={() => kind && onSave({ kind, note: note.trim() || undefined })}
          >
            Attach flag
          </button>
        </div>
      </div>
    </div>
  );
}
