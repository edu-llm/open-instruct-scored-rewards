'use client';

import { useState } from 'react';

export function StartButton({ src }: { src: string | null }) {
  const [state, setState] = useState<'idle' | 'working' | 'failed'>('idle');

  async function start() {
    setState('working');
    const res = await fetch('/api/session', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(src ? { src } : {}),
    });
    if (!res.ok) {
      setState('failed');
      return;
    }
    window.location.href = '/how';
  }

  return (
    <>
      <button className="primary" disabled={state === 'working'} onClick={() => void start()}>
        {state === 'working' ? 'Starting…' : 'Start'}
      </button>
      {state === 'failed' ? (
        <span className="notice bad">
          Could not start a session. If you are using a strict privacy extension, it may be blocking
          the cookie this needs.
        </span>
      ) : null}
    </>
  );
}
