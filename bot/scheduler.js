/**
 * scheduler.js
 *
 * Runs monthly leaderboard summary on the last day of each month.
 * Call startScheduler(client, groupJid) from index.js once the bot is ready.
 *
 * Usage in index.js:
 *   const { startScheduler } = require('./scheduler');
 *   client.on('ready', () => {
 *     startScheduler(client, process.env.GROUP_JID);
 *   });
 */

const { getMonthlyLeaderboard, archiveMonth } = require('./db');
const { formatMonthlyWinner }                 = require('./formatter');

function startScheduler(client, groupJid) {
  if (!groupJid) {
    console.warn('[scheduler] GROUP_JID not set — monthly summaries disabled');
    return;
  }

  console.log(`[scheduler] Started — will post monthly summary to ${groupJid}`);

  // Check every hour whether it's time to post
  setInterval(() => _tick(client, groupJid), 60 * 60 * 1000);

  // Also check immediately on startup in case bot was down at month end
  _tick(client, groupJid);
}

let _lastPostedMonth = null;

async function _tick(client, groupJid) {
  const now   = new Date();
  const isLastDay = _isLastDayOfMonth(now);
  const isEvening = now.getHours() >= 20;  // post at 8pm on last day

  if (!isLastDay || !isEvening) return;

  const yearMonth = now.toISOString().slice(0, 7);
  if (_lastPostedMonth === yearMonth) return;  // already posted this month

  try {
    await _postMonthlySummary(client, groupJid, yearMonth);
    archiveMonth(yearMonth);
    _lastPostedMonth = yearMonth;
    console.log(`[scheduler] Posted monthly summary for ${yearMonth}`);
  } catch (err) {
    console.error('[scheduler] Failed to post summary:', err.message);
  }
}

async function _postMonthlySummary(client, groupJid, yearMonth) {
  const rows    = getMonthlyLeaderboard(yearMonth);
  const message = formatMonthlyWinner(rows, yearMonth);

  if (!message) {
    await client.sendMessage(groupJid, `📅 *${yearMonth}* — no pints scored this month. Disgraceful.`);
    return;
  }

  await client.sendMessage(groupJid, message);
}

function _isLastDayOfMonth(date) {
  const tomorrow = new Date(date);
  tomorrow.setDate(tomorrow.getDate() + 1);
  return tomorrow.getDate() === 1;
}

// Manual trigger — useful for testing
// Call with: node -e "require('./scheduler')._postNow(client, jid, '2025-01')"
async function _postNow(client, groupJid, yearMonth) {
  await _postMonthlySummary(client, groupJid, yearMonth);
}

module.exports = { startScheduler, _postNow };