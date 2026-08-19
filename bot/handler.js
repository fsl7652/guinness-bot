/**
 * handler.js
 *
 * Routes incoming WhatsApp messages to the correct command handler.
 * Responsibilities:
 *   - Parse command from message text / caption
 *   - Rate limiting (per user, per command)
 *   - Download and save images (raw training data + named pint/splitg)
 *   - Dispatch to score / splitg / leaderboard / me / ping / help
 *   - Reply with formatted messages from formatter.js
 */

const path = require('path');
const fs   = require('fs');
const pino = require('pino');

const { scoreImage }     = require('./scorer');
const { db, queries }    = require('./db');
const { format }         = require('./formatter');
const { saveChat }       = require('./scheduler');

const logger   = pino({ level: process.env.LOG_LEVEL || 'info' });
const DATA_DIR = path.join(__dirname, '..', 'data', 'raw');
fs.mkdirSync(DATA_DIR, { recursive: true });

// ── Rate limiting ─────────────────────────────────────────────
// In-memory per-user per-command cooldowns.
// Resets on restart — good enough for a pub bot.

const RATE_LIMITS = {
  '!score':       60_000,   // 1 min between scores (model is slow)
  '!splitg':      30_000,   // 30 s between splitg checks
  '!leaderboard': 10_000,   // 10 s
  '!me':          10_000,
  '!ping':         5_000,
};

// Map of `userId:cmd` → timestamp of last use
const lastUsed = new Map();

function isRateLimited(userId, cmd) {
  const limit = RATE_LIMITS[cmd];
  if (!limit) return false;

  const key  = `${userId}:${cmd}`;
  const last = lastUsed.get(key) ?? 0;
  const now  = Date.now();

  if (now - last < limit) {
    return Math.ceil((limit - (now - last)) / 1000);  // seconds remaining
  }

  lastUsed.set(key, now);
  return false;
}

// ── Image helpers ─────────────────────────────────────────────

async function downloadAndSave(sock, msg, downloadMediaMessage, type) {
  const buffer = await downloadMediaMessage(
    msg, 'buffer', {},
    { logger, reuploadRequest: sock.updateMediaMessage }
  );

  const userId   = senderOf(msg).split('@')[0].replace(/\W/g, '_');
  const filename = `${type}_${userId}_${Date.now()}.jpg`;
  const filepath = path.join(DATA_DIR, filename);

  await fs.promises.writeFile(filepath, buffer);
  logger.info({ filepath }, '💾 Image saved');

  // Record in DB for training data tracking
  queries.insertImage.run(userId, filepath, type, new Date().toISOString());

  return filepath;
}

// ── Message parsing ───────────────────────────────────────────

function extractText(msg) {
  return (
    msg.message?.conversation              ||
    msg.message?.extendedTextMessage?.text ||
    msg.message?.imageMessage?.caption     ||
    ''
  ).trim();
}

function senderOf(msg) {
  return msg.key.participant || msg.key.remoteJid;
}

function hasImage(msg) {
  return !!msg.message?.imageMessage;
}

// ── Main router ───────────────────────────────────────────────

async function routeMessage(sock, msg, { downloadMediaMessage }) {
  const jid      = msg.key.remoteJid;
  const sender   = senderOf(msg);
  const userId   = sender.split('@')[0];
  const pushName = msg.pushName || 'Unknown';
  const text     = extractText(msg);

  if (!text.startsWith('!')) return;

  const [rawCmd, ...args] = text.split(/\s+/);
  const cmd = rawCmd.toLowerCase();

  logger.info({ cmd, userId, hasImg: hasImage(msg) }, '⌨️ Command');

  // Register this chat so the scheduler can broadcast to it
  saveChat(jid);

  // Rate limit check (skip for unknown commands)
  const wait = isRateLimited(userId, cmd);
  if (wait) {
    await sock.sendMessage(jid, {
      text: `⏳ Slow down! Try \`${cmd}\` again in ${wait}s.`,
    });
    return;
  }

  switch (cmd) {
    case '!score':    return handleScore(sock, msg, jid, sender, userId, pushName, downloadMediaMessage);
    case '!splitg':   return handleSplitg(sock, msg, jid, sender, userId, pushName, downloadMediaMessage);
    case '!leaderboard':
    case '!lb':       return handleLeaderboard(sock, jid);
    case '!me':       return handleMe(sock, jid, userId, pushName);
    case '!ping':     return sock.sendMessage(jid, { text: '🏓 Pong!' });
    case '!help':     return handleHelp(sock, jid);
    default:
      logger.debug({ cmd }, 'Unknown command');
  }
}

// ── !score ────────────────────────────────────────────────────

async function handleScore(sock, msg, jid, sender, userId, pushName, downloadMediaMessage) {
  if (!hasImage(msg)) {
    return sock.sendMessage(jid, {
      text: '📸 Attach a pint photo and send *!score* as the caption.',
    });
  }

  await sock.sendMessage(jid, { text: '🍺 Analysing your pint…' });

  let filepath;
  try {
    filepath = await downloadAndSave(sock, msg, downloadMediaMessage, 'pint');
    const result = await scoreImage(filepath, null, pushName);

    // Persist each detected glass
    for (const glass of result.glasses) {
      queries.insertScore.run(
        userId, pushName, filepath,
        glass.pint_score,
        JSON.stringify(glass.breakdown ?? {}),
        JSON.stringify(glass.splitg    ?? {}),
        JSON.stringify(glass.warnings  ?? []),
        'pint'
      );
    }

    await sock.sendMessage(jid, { text: result.message });

  } catch (err) {
    logger.error({ err, filepath }, '❌ !score failed');
    await sock.sendMessage(jid, { text: friendlyError(err.message) });
  }
}

// ── !splitg ───────────────────────────────────────────────────

async function handleSplitg(sock, msg, jid, sender, userId, pushName, downloadMediaMessage) {
  if (!hasImage(msg)) {
    return sock.sendMessage(jid, {
      text: '📸 Attach a mid-sip photo and send *!splitg* as the caption.',
    });
  }

  await sock.sendMessage(jid, { text: '🔍 Checking the G split…' });

  let filepath;
  try {
    filepath = await downloadAndSave(sock, msg, downloadMediaMessage, 'splitg');
    const result = await scoreImage(null, filepath, pushName);

    // Find the most recent score for this user to attach splitg result to
    const lastScore = queries.lastScoreForUser.get(userId);
    if (lastScore) {
      queries.updateSplitg.run(
        JSON.stringify(result.splitg ?? {}),
        lastScore.id
      );
    }

    await sock.sendMessage(jid, { text: format.splitgReply(result.splitg, pushName) });

  } catch (err) {
    logger.error({ err, filepath }, '❌ !splitg failed');
    await sock.sendMessage(jid, { text: friendlyError(err.message) });
  }
}

// ── !leaderboard ──────────────────────────────────────────────

async function handleLeaderboard(sock, jid) {
  const rows = queries.leaderboard.all(10);
  await sock.sendMessage(jid, { text: format.leaderboard(rows) });
}

// ── !me ───────────────────────────────────────────────────────

async function handleMe(sock, jid, userId, pushName) {
  const stats   = queries.userStats.get(userId);
  const rankRow = queries.userRank.get(userId);
  const recent  = queries.recentScores.all(userId, 3);
  await sock.sendMessage(jid, { text: format.me(pushName, stats, rankRow, recent) });
}

// ── !help ─────────────────────────────────────────────────────

async function handleHelp(sock, jid) {
  await sock.sendMessage(jid, { text: format.help() });
}

// ── Error messages ────────────────────────────────────────────

function friendlyError(msg) {
  if (msg.includes('No glasses detected'))
    return '⚠️ No pint glass detected — make sure the glass is clearly visible and try again.';
  if (msg.includes('timed out'))
    return '⚠️ Scoring timed out. Try again in a moment.';
  if (msg.includes('decode') || msg.includes('imdecode'))
    return '⚠️ Couldn\'t read the image. Send a standard JPEG photo.';
  return '⚠️ Something went wrong. Please try again.';
}

module.exports = { routeMessage };
