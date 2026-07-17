const fs   = require('fs');
const path = require('path');

const { scoreImage }                                    = require('./scorer');
const { saveScore, getLeaderboard, getUserStats,
        getRecentScores, currentYearMonth,
        getMonthlyLeaderboard }                         = require('./db');
const { formatScore, formatLeaderboard,
        formatUserStats }                               = require('./formatter');

const DATA_DIR = path.join(__dirname, '../data/raw');

// Per-user rate limit — 1 score per hour
const RATE_LIMIT  = new Map();
const RATE_MS     = 60 * 60 * 1000;

// Pending split-the-G: jid → { pintBase64, timestamp }
const PENDING_SPLITG  = new Map();
const SPLITG_TIMEOUT  = 5 * 60 * 1000;

// ── Main handler ──────────────────────────────────────────────

async function handleMessage(msg, client) {
  try {
    await _route(msg, client);
  } catch (err) {
    console.error('[handler]', err);
    try { await client.sendMessage(msg.from, '❌ Something went wrong — try again'); }
    catch (_) {}
  }
}

async function _route(msg, client) {
  const body = msg.body?.trim().toLowerCase() ?? '';

  if (body === '!ping') {
    return client.sendMessage(msg.from, '🍺 Still pouring');
  }

  if (body === '!help') {
    return client.sendMessage(msg.from, HELP_TEXT);
  }

  if (body === '!leaderboard') {
    const rows = getLeaderboard();
    return client.sendMessage(msg.from, formatLeaderboard(rows));
  }

  if (body === '!monthly') {
    const rows = getMonthlyLeaderboard(currentYearMonth());
    return client.sendMessage(msg.from,
      formatLeaderboard(rows, `🍺 *${currentYearMonth()} Leaderboard*`)
    );
  }

  if (body === '!me') {
    const contact     = await msg.getContact();
    const displayName = contact.pushname || contact.number;
    const stats       = getUserStats(msg.from);
    const recent      = getRecentScores(msg.from);
    return client.sendMessage(msg.from, formatUserStats(stats, recent, displayName));
  }

  if (!msg.hasMedia) return;

  if (body === '!splitg') {
    const pending = PENDING_SPLITG.get(msg.from);
    if (!pending) {
      return client.sendMessage(msg.from,
        '❌ No pending pint — score one first with !score then send your mid-sip photo with !splitg'
      );
    }
    if (Date.now() - pending.timestamp > SPLITG_TIMEOUT) {
      PENDING_SPLITG.delete(msg.from);
      return client.sendMessage(msg.from, '⏱ Split-the-G window expired — score another pint first');
    }

    const media = await _downloadMedia(msg, client);
    if (!media) return;

    await client.sendMessage(msg.from, '🔍 Checking the G split...');

    const contact     = await msg.getContact();
    const displayName = contact.pushname || contact.number;
    const result      = await scoreImage(pending.pintBase64, media.data, displayName);

    PENDING_SPLITG.delete(msg.from);
    await _saveAndReply(msg, client, result, displayName);
    return;
  }

  if (body !== '!score') return;

  const lastScore = RATE_LIMIT.get(msg.from);
  if (lastScore && Date.now() - lastScore < RATE_MS) {
    const mins = Math.ceil((RATE_MS - (Date.now() - lastScore)) / 60000);
    return client.sendMessage(msg.from, `⏱ One score per hour — try again in ${mins} min`);
  }

  const media = await _downloadMedia(msg, client);
  if (!media) return;

  _saveImage(media.data, msg.from);

  await client.sendMessage(msg.from, '🔍 Judging your pint...');

  const contact     = await msg.getContact();
  const displayName = contact.pushname || contact.number;

  const result = await scoreImage(media.data, null, displayName);

  RATE_LIMIT.set(msg.from, Date.now());
  PENDING_SPLITG.set(msg.from, { pintBase64: media.data, timestamp: Date.now() });

  await _saveAndReply(msg, client, result, displayName);

  await client.sendMessage(msg.from,
    '💧 Want to check your G split? Send a mid-sip photo with *!splitg* within 5 minutes'
  );
}


async function _downloadMedia(msg, client) {
  let media;
  
  try {
    // Try getting raw message data directly from the page
    const rawMsg = await client.pupPage.evaluate(async (msgId) => {
      const msg = window.Store.Msg.get(msgId);
      if (!msg) return null;
      const mediaData = await window.Store.DownloadManager.downloadAndMaybeDecrypt({
        directPath: msg.directPath,
        encFilehash: msg.encFilehash,
        filehash: msg.filehash,
        mediaKey: msg.mediaKey,
        type: msg.type,
        signal: new AbortController().signal
      });
      return {
        data: btoa(String.fromCharCode(...new Uint8Array(mediaData))),
        mimetype: msg.mimetype
      };
    }, msg.id._serialized);

    if (rawMsg) {
      media = rawMsg;
    } else {
      throw new Error('null from page evaluate');
    }
  } catch (err) {
    console.error('[handler] all download methods failed:', err.message);
    await client.sendMessage(msg.from, '❌ Could not download image — try again');
    return null;
  }

  if (!media?.mimetype?.startsWith('image/')) {
    await client.sendMessage(msg.from, '❌ Send an image file');
    return null;
  }

  return media;
}

function _saveImage(base64Data, jid) {
  try {
    const name = `${Date.now()}_${jid.replace(/[^a-zA-Z0-9]/g, '_')}.jpg`;
    fs.writeFileSync(path.join(DATA_DIR, name), Buffer.from(base64Data, 'base64'));
    return name;
  } catch (err) {
    console.error('[handler] Failed to save image:', err.message);
    return null;
  }
}

async function _saveAndReply(msg, client, result, displayName) {
  const primary = result.glasses?.[0] ?? result;
  const imgFile = _saveImage(
    result.glasses?.[0] ? null : null, msg.from
  );
  saveScore(msg.from, displayName, primary, imgFile);
  await client.sendMessage(msg.from, formatScore(result, displayName));
}

const HELP_TEXT = [
  '🍺 *Guinness Bot Commands*',
  '',
  '*!score*  + pint photo     → score your pint',
  '*!splitg* + mid-sip photo  → check the G split (within 5 min of !score)',
  '*!leaderboard*             → all-time top 10',
  '*!monthly*                 → this month\'s leaderboard',
  '*!me*                      → your personal stats',
  '*!ping*                    → check bot is alive',
  '',
  '_Tips: one pint per photo, glass centred in frame_',
].join('\n');

module.exports = { handleMessage };