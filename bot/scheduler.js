/**
 * scheduler.js
 *
 * Cron jobs:
 *
 *   Monthly (last day of month, 21:00)
 *     1. Post the final leaderboard to all active chats
 *     2. Archive the month's scores to scores_archive
 *     3. Reset the live scores table
 *
 *   Daily (09:00) — optional nudge if no submissions in 48h
 *     Posts a gentle reminder to active chats.
 *     Set DAILY_NUDGE=false in .env to disable.
 *
 * Active chats are tracked in data/chats.json (written by handler.js
 * on first command in any chat). The scheduler posts to all of them.
 *
 * Uses a minimal cron implementation (no external dep) — setInterval
 * wakes every minute and checks if it's time to fire.
 */

'use strict';

const fs   = require('fs');
const path = require('path');
const pino = require('pino');

const { queries, archiveMonth } = require('./db');
const { format }                = require('./formatter');

const logger     = pino({ level: process.env.LOG_LEVEL || 'info' });
const CHATS_FILE = path.join(__dirname, '..', 'data', 'chats.json');
const NUDGE      = process.env.DAILY_NUDGE !== 'false';

// ── Active chat registry ──────────────────────────────────────

function loadChats() {
  try {
    return JSON.parse(fs.readFileSync(CHATS_FILE, 'utf8'));
  } catch {
    return [];
  }
}

function saveChat(jid) {
  const chats = loadChats();
  if (!chats.includes(jid)) {
    chats.push(jid);
    fs.mkdirSync(path.dirname(CHATS_FILE), { recursive: true });
    fs.writeFileSync(CHATS_FILE, JSON.stringify(chats, null, 2));
  }
}

// ── Broadcast helpers ─────────────────────────────────────────

async function broadcast(sock, text) {
  const chats = loadChats();
  logger.info({ chats: chats.length }, '📣 Broadcasting');

  for (const jid of chats) {
    try {
      await sock.sendMessage(jid, { text });
      // Small delay between messages to avoid rate-limiting
      await new Promise(r => setTimeout(r, 500));
    } catch (err) {
      logger.warn({ err, jid }, '⚠️ Broadcast failed for chat');
    }
  }
}

// ── Monthly job ───────────────────────────────────────────────

async function runMonthlyJob(sock) {
  const now   = new Date();
  const month = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;

  logger.info({ month }, '📅 Running monthly job');

  try {
    // 1. Grab current month's leaderboard before archiving
    const rows   = queries.leaderboard.all(10);
    const winner = rows[0] ?? null;

    // 2. Post final standings
    const msg = format.monthlyArchive(month, rows, winner);
    await broadcast(sock, msg);

    // 3. Archive and reset
    const count = archiveMonth();
    logger.info({ month, archivedUsers: count }, '✅ Month archived');

    // 4. Post new-month opener
    await new Promise(r => setTimeout(r, 2_000));
    const nextMonth = new Date(now.getFullYear(), now.getMonth() + 1)
      .toLocaleString('en-GB', { month: 'long' });
    await broadcast(sock, `🍺 *${nextMonth} is here. New month, fresh leaderboard.*\nSend your first pint with !score.`);

  } catch (err) {
    logger.error({ err }, '❌ Monthly job failed');
  }
}

// ── Daily nudge ───────────────────────────────────────────────

async function runDailyNudge(sock) {
  if (!NUDGE) return;

  // Only nudge if there's been at least one score ever but none in 48h
  const cutoff = new Date(Date.now() - 48 * 3_600_000).toISOString();
  const db     = require('./db').db;
  const recent = db.prepare(
    `SELECT COUNT(*) AS n FROM scores WHERE scored_at > ? AND score_type = 'pint'`
  ).get(cutoff);

  if (recent.n > 0) return;  // someone scored recently — no nudge

  const total = db.prepare(`SELECT COUNT(*) AS n FROM scores WHERE score_type = 'pint'`).get();
  if (total.n === 0) return;  // no scores at all yet — not started

  logger.info('💬 Sending daily nudge');

  const nudges = [
    '🍺 No pints scored in 48 hours. The barman is waiting. !score',
    '👀 Leaderboard looking quiet. Someone pour a pint. !score',
    '🏆 Top spot is up for grabs. Send a pint photo to claim it. !score',
  ];

  const msg = nudges[Math.floor(Math.random() * nudges.length)];
  await broadcast(sock, msg);
}

// ── Cron loop ─────────────────────────────────────────────────

let interval = null;
let lastMonthlyFired = null;
let lastNudgeFired   = null;

function start(sock) {
  if (interval) return;

  logger.info('🕐 Scheduler started');

  interval = setInterval(async () => {
    const now = new Date();
    const hh  = now.getHours();
    const mm  = now.getMinutes();
    const dd  = now.getDate();

    // Last day of month check — fire at 21:00
    const isLastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0).getDate() === dd;
    const monthKey  = `${now.getFullYear()}-${now.getMonth()}`;

    if (isLastDay && hh === 21 && mm === 0 && lastMonthlyFired !== monthKey) {
      lastMonthlyFired = monthKey;
      await runMonthlyJob(sock);
    }

    // Daily nudge at 09:00
    const dayKey = now.toDateString();
    if (hh === 9 && mm === 0 && lastNudgeFired !== dayKey) {
      lastNudgeFired = dayKey;
      await runDailyNudge(sock);
    }

  }, 60_000); // tick every minute
}

function stop() {
  if (interval) {
    clearInterval(interval);
    interval = null;
    logger.info('🕐 Scheduler stopped');
  }
}

// Expose saveChat so handler.js can register new chats
module.exports = { start, stop, saveChat };
