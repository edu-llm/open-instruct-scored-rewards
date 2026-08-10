import { initBotId } from 'botid/client/core';

// Deep Analysis is billed per checkBotId() call, so only the two low-volume routes are protected:
// ~1 session per rater and ~1 assignment per 12 labels. /api/label is the hot path and is covered
// by the session cookie and the WAF instead.
initBotId({
  protect: [
    { path: '/api/session', method: 'POST' },
    { path: '/api/assignment', method: 'POST' },
  ],
});
