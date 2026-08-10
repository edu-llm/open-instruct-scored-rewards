'use client';

import { useCallback, useEffect, useState } from 'react';
import { ThemeToggle } from './ThemeToggle';
import type { AdminStats, RaterQcRow } from '@/lib/types';

const KEY = 'tl_admin_token';

function n(value: number | null, digits = 2): string {
  return value === null ? '—' : value.toFixed(digits);
}

export function AdminDashboard() {
  const [token, setToken] = useState('');
  const [entered, setEntered] = useState('');
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [raters, setRaters] = useState<RaterQcRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setToken(sessionStorage.getItem(KEY) ?? '');
  }, []);

  const load = useCallback(async (bearer: string) => {
    setError(null);
    const headers = { authorization: `Bearer ${bearer}` };
    const [s, r] = await Promise.all([
      fetch('/api/admin/stats', { headers }),
      fetch('/api/admin/raters?sort=labels', { headers }),
    ]);
    if (!s.ok) {
      setError(s.status === 401 ? 'That token was not accepted.' : `Server said ${s.status}.`);
      return;
    }
    setStats((await s.json()) as AdminStats);
    setRaters(r.ok ? ((await r.json()) as RaterQcRow[]) : []);
  }, []);

  useEffect(() => {
    if (token) void load(token);
  }, [token, load]);

  if (!token) {
    return (
      <div className="page" style={{ maxWidth: 460 }}>
        <h1>Admin</h1>
        <p>Paste ADMIN_TOKEN. It is kept in this tab only.</p>
        <input
          type="text"
          value={entered}
          placeholder="bearer token"
          onChange={(e) => setEntered(e.target.value)}
        />
        <div className="row" style={{ marginTop: 14 }}>
          <button
            className="primary"
            onClick={() => {
              sessionStorage.setItem(KEY, entered.trim());
              setToken(entered.trim());
            }}
          >
            Open
          </button>
        </div>
        {error ? <p className="notice bad">{error}</p> : null}
      </div>
    );
  }

  return (
    <div className="page" style={{ maxWidth: 1180 }}>
      <div className="row">
        <span className="eyebrow" style={{ margin: 0 }}>
          Admin · batch {stats?.batchId ?? '…'}
        </span>
        <span className="spacer" />
        <ThemeToggle />
        <button
          className="quiet"
          onClick={() => {
            sessionStorage.removeItem(KEY);
            setToken('');
            setStats(null);
          }}
        >
          Sign out
        </button>
      </div>

      {error ? <p className="notice bad">{error}</p> : null}
      {!stats ? <h1>Loading…</h1> : null}

      {stats ? (
        <>
          <h1>
            {stats.labels.total.toLocaleString()} labels
            <span className="meta" style={{ marginLeft: 12, fontSize: 15 }}>
              {stats.labels.last24h.toLocaleString()} in the last day
            </span>
          </h1>
          <div className="row" style={{ marginBottom: 8 }}>
            <span className="chip">{stats.raters.total} raters</span>
            <span className="chip good">{stats.raters.active} active</span>
            <span className="chip warn">{stats.raters.probation} probation</span>
            <span className="chip bad">{stats.raters.suspended} suspended</span>
            <span className="chip">{stats.raters.failed} failed calibration</span>
          </div>

          {stats.alerts.length > 0 ? (
            <>
              <h2>The rubric is on trial too</h2>
              {stats.alerts.map((a) => (
                <div key={a} className="notice bad" style={{ marginBottom: 8 }}>
                  {a}
                </div>
              ))}
            </>
          ) : null}

          <h2>Progress by metric</h2>
          <table>
            <thead>
              <tr>
                <th>Metric</th>
                <th style={{ textAlign: 'right' }}>Items</th>
                <th style={{ textAlign: 'right' }}>Got / target</th>
                <th style={{ textAlign: 'right' }}>Pairs done</th>
                <th style={{ textAlign: 'right' }}>kappa_w</th>
                <th style={{ textAlign: 'right' }}>flag</th>
                <th style={{ textAlign: 'right' }}>n/a</th>
                <th style={{ textAlign: 'right' }}>length rho</th>
              </tr>
            </thead>
            <tbody>
              {stats.progress.map((p) => {
                const qc = stats.metricQc.find((m) => m.metricKey === p.metricKey);
                return (
                  <tr key={p.metricKey}>
                    <td className="ink mono">{p.metricKey}</td>
                    <td className="num">{p.nItems}</td>
                    <td className="num">
                      {p.got} / {p.target}
                    </td>
                    <td className="num">
                      {p.pairsDone} / {p.pairsPlanned}
                    </td>
                    <td className="num">{n(qc?.kappaW ?? null)}</td>
                    <td className="num">{n(qc?.flagRate ?? null)}</td>
                    <td className="num">{n(qc?.naRate ?? null)}</td>
                    <td className="num">{n(qc?.lengthRho ?? null)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>

          <h2>Raters</h2>
          <p style={{ fontSize: 14 }}>
            Gold accuracy and kappa are the two that carry weight. The rest are screens that put a
            rater in front of a human, not verdicts.
          </p>
          <table>
            <thead>
              <tr>
                <th>Handle</th>
                <th>Status</th>
                <th>Cohort</th>
                <th style={{ textAlign: 'right' }}>Labels</th>
                <th style={{ textAlign: 'right' }}>gold acc</th>
                <th style={{ textAlign: 'right' }}>kappa_w</th>
                <th style={{ textAlign: 'right' }}>median ms</th>
                <th style={{ textAlign: 'right' }}>&lt; floor</th>
                <th style={{ textAlign: 'right' }}>mode+</th>
                <th style={{ textAlign: 'right' }}>runs z</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {raters.map((r) => (
                <tr key={r.raterId}>
                  <td className="ink mono">{r.handle}</td>
                  <td>{r.status}</td>
                  <td>{r.cohort ?? '—'}</td>
                  <td className="num">{r.nLabels}</td>
                  <td className="num">{n(r.goldAcc)}</td>
                  <td className="num">{n(r.kappaW)}</td>
                  <td className="num">{r.medianDwellMs ?? '—'}</td>
                  <td className="num">{n(r.fracBelowFloor)}</td>
                  <td className="num">{n(r.modeShareExcess)}</td>
                  <td className="num">{n(r.runsZ, 1)}</td>
                  <td style={{ textAlign: 'right' }}>
                    <button
                      className="quiet"
                      onClick={async () => {
                        const reason = window.prompt(`Suspend ${r.handle}. Reason?`);
                        if (!reason) return;
                        await fetch(`/api/admin/rater/${r.raterId}`, {
                          method: 'PATCH',
                          headers: {
                            authorization: `Bearer ${token}`,
                            'content-type': 'application/json',
                          },
                          body: JSON.stringify({ status: 'suspended', reason }),
                        });
                        void load(token);
                      }}
                    >
                      suspend
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <h2>Export</h2>
          <p style={{ fontSize: 14 }}>
            Labels are never deleted and status is never applied retroactively, so excluding a
            cohort is a filter here rather than a change to the data.
          </p>
          <div className="row">
            {['active', 'probation', 'any'].map((f) => (
              <button
                key={f}
                onClick={async () => {
                  const res = await fetch(`/api/admin/export?min_status=${f}`, {
                    headers: { authorization: `Bearer ${token}` },
                  });
                  if (!res.ok) {
                    setError(`Export failed: ${res.status}`);
                    return;
                  }
                  const blob = await res.blob();
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url;
                  a.download = `export-${f}.json`;
                  a.click();
                  URL.revokeObjectURL(url);
                }}
              >
                min_status={f}
              </button>
            ))}
          </div>
        </>
      ) : null}
    </div>
  );
}
