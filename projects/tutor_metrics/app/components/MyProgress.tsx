'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ThemeToggle } from './ThemeToggle';
import type { RaterState } from '@/lib/types';

const BAND: Record<RaterState['goldBand'], { label: string; cls: string }> = {
  good: { label: 'Most checks correct', cls: 'chip good' },
  mixed: { label: 'Some checks missed', cls: 'chip warn' },
  unknown: { label: 'Not enough checks yet', cls: 'chip' },
};

export function MyProgress() {
  const [me, setMe] = useState<RaterState | null>(null);
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    void (async () => {
      const res = await fetch('/api/me');
      if (!res.ok) {
        setMissing(true);
        return;
      }
      setMe((await res.json()) as RaterState);
    })();
  }, []);

  if (missing) {
    return (
      <div className="page">
        <h1>No session on this device</h1>
        <Link href="/">
          <button className="primary">Start</button>
        </Link>
      </div>
    );
  }
  if (!me) return <div className="page"><h1>Loading…</h1></div>;

  return (
    <div className="page">
      <div className="row">
        <span className="eyebrow" style={{ margin: 0 }}>
          Your progress
        </span>
        <span className="spacer" />
        <ThemeToggle />
      </div>

      <h1 className="mono" style={{ fontSize: 26 }}>
        {me.handle}
      </h1>
      <div className="row" style={{ marginBottom: 20 }}>
        <span className="chip">{me.status}</span>
        <span className={BAND[me.goldBand].cls}>{BAND[me.goldBand].label}</span>
        <span className="meta">{me.nLabels} labels</span>
      </div>

      <div className="notice">
        We do not show you how your answers compare with other raters while collection is open.
        Seeing it would turn the task from <em>read the anchors and answer</em> into <em>guess what
        the others said</em>, and inter-rater agreement is the one number this project needs to be
        able to trust. It is published to everyone once the round closes.
      </div>

      {me.metrics.length > 0 ? (
        <>
          <h2>By question</h2>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th style={{ textAlign: 'right' }}>Labels</th>
                <th>Warm-up</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {me.metrics.map((m) => (
                <tr key={m.key}>
                  <td className="ink mono">{m.key}</td>
                  <td className="num">{m.nLabels}</td>
                  <td>{m.primed ? 'done' : '—'}</td>
                  <td style={{ textAlign: 'right' }}>
                    {me.qualified ? (
                      <Link href={`/label?metric=${encodeURIComponent(m.key)}`}>switch to this</Link>
                    ) : null}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : null}

      <div className="row" style={{ marginTop: 28 }}>
        {me.qualified ? (
          <Link href="/label">
            <button className="primary">Keep labelling</button>
          </Link>
        ) : (
          <Link href="/calibration">
            <button className="primary">Go to calibration</button>
          </Link>
        )}
      </div>
    </div>
  );
}
