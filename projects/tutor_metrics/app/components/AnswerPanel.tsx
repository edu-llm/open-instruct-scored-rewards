'use client';

import type { Anchor, Metric, NaReason } from '@/lib/types';

export const NA_REASONS: Array<{ key: NaReason; label: string }> = [
  { key: 'missing_reference', label: 'The worked solution this needs is not shown' },
  { key: 'missing_context', label: 'The earlier dialogue this needs is not shown' },
  { key: 'not_a_tutor_turn', label: 'This is not a tutor turn' },
  { key: 'unintelligible', label: 'The turn is unintelligible' },
];

/** A yes/no metric shows the words as well as the numeral; a 5-rung one shows the rung name. */
function binaryTag(metric: Metric, anchor: Anchor): string | null {
  if (metric.hi - metric.lo !== 1) return null;
  return anchor.value === metric.lo ? 'No' : 'Yes';
}

interface AnchorProps {
  metric: Metric;
  selected: number | null;
  disabled: boolean;
  onPick: (value: number) => void;
}

export function AnchorList({ metric, selected, disabled, onPick }: AnchorProps) {
  return (
    <div className="anchors">
      {metric.anchors.map((a) => {
        const tag = a.name ?? binaryTag(metric, a);
        return (
          <button
            key={a.value}
            type="button"
            className={`anchor${selected === a.value ? ' sel' : ''}`}
            disabled={disabled}
            aria-pressed={selected === a.value}
            onClick={() => onPick(a.value)}
          >
            <kbd>{a.value}</kbd>
            <span>
              {tag ? <span className="tag">{tag}</span> : null}
              <span className="text">{a.label}</span>
            </span>
            <span className="tick" aria-hidden>
              ✓
            </span>
          </button>
        );
      })}
    </div>
  );
}

/**
 * assistance_level. Five ordered rungs of Graesser's ladder, drawn as a ladder because the ORDER
 * is load-bearing: the reward gives partial credit for an adjacent rung, so a rater who reads the
 * list as five unordered categories is answering a different question.
 */
export function RungLadder({ metric, selected, disabled, onPick }: AnchorProps) {
  return (
    <div>
      <div className="ladder-axis">
        <span>Least tutor control</span>
        <span className="track" aria-hidden />
        <span>Most</span>
      </div>
      <div className="ladder">
        {metric.anchors.map((a) => (
          <button
            key={a.value}
            type="button"
            className={`rung${selected === a.value ? ' sel' : ''}`}
            disabled={disabled}
            aria-pressed={selected === a.value}
            onClick={() => onPick(a.value)}
          >
            <kbd>{a.value}</kbd>
            <span className="rail" aria-hidden>
              <span className="dot" />
            </span>
            <span>
              <span className="name">{a.name ?? `Level ${a.value}`}</span>
              <span className="text">{a.label}</span>
            </span>
            <span className="tick" aria-hidden>
              ✓
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}

interface SpanProps {
  count: number;
  chosen: number | null;
  onPick: (index: number) => void;
  onCancel: () => void;
}

/**
 * Two keystrokes total: the score, then the sentence. Typing a quotation is far too much friction
 * at three hundred items an hour, and the required span is the single rule most responsible for
 * the difference between the two reliability regimes.
 */
export function SpanPrompt({ count, chosen, onPick, onCancel }: SpanProps) {
  return (
    <div className="subpanel">
      <h4>Which part does it?</h4>
      <div className="pick-row">
        {Array.from({ length: count }, (_, i) => (
          <button
            key={i}
            type="button"
            className={`pick${chosen === i ? ' sel' : ''}`}
            onClick={() => onPick(i)}
          >
            <kbd>{i + 1}</kbd>
            <span>sentence {i + 1}</span>
          </button>
        ))}
      </div>
      <div className="hints">
        <span>or drag to select any part of the turn</span>
        <button type="button" className="quiet" onClick={onCancel}>
          <kbd>esc</kbd> change the score
        </button>
      </div>
    </div>
  );
}

interface NaProps {
  onPick: (reason: NaReason) => void;
  onCancel: () => void;
}

export function NaPrompt({ onPick, onCancel }: NaProps) {
  return (
    <div className="subpanel">
      <h4>Why does this not apply?</h4>
      <div className="pick-row" style={{ flexDirection: 'column' }}>
        {NA_REASONS.map((r, i) => (
          <button key={r.key} type="button" className="pick" onClick={() => onPick(r.key)}>
            <kbd>{i + 1}</kbd>
            <span>{r.label}</span>
          </button>
        ))}
      </div>
      <div className="hints">
        <button type="button" className="quiet" onClick={onCancel}>
          <kbd>esc</kbd> back
        </button>
      </div>
    </div>
  );
}
