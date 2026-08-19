/**
 * formatter.js
 *
 * Converts result objects into WhatsApp message strings.
 * All user-facing text lives here — nothing formats messages elsewhere.
 *
 * Exports: format.scoreReply, format.splitgReply, format.leaderboard,
 *          format.me, format.help, format.monthlyArchive
 */

'use strict';

// ── Helpers ───────────────────────────────────────────────────

function stars(score) {
  const filled = Math.round(score / 2);
  return '⭐'.repeat(filled) + '☆'.repeat(5 - filled);
}

function tryParse(json) {
  try { return typeof json === 'string' ? JSON.parse(json) : json; }
  catch { return null; }
}

function bar(score, width = 10) {
  const filled = Math.round((score / 10) * width);
  return '█'.repeat(filled) + '░'.repeat(width - filled);
}

function pct(n) {
  return n != null ? `${Math.round(n * 100)}%` : '—';
}

// ── Score reply ───────────────────────────────────────────────

/**
 * Full pint score reply.
 * Used when infer.py doesn't return a pre-formatted message,
 * or when we want JS-controlled formatting.
 *
 * @param {string}  name    - display name
 * @param {object}  glass   - single glass result from infer.py
 * @param {number}  [index] - 1-based index if multiple glasses
 */
function scoreReply(name, glass, index = null) {
  const bd    = tryParse(glass.breakdown) ?? {};
  const sg    = tryParse(glass.splitg)    ?? {};
  const warns = tryParse(glass.warnings)  ?? [];
  const score = glass.pint_score ?? glass.final ?? 0;

  const header = index
    ? `🍺 *${name}'s Glass ${index} — ${score}/10*`
    : `🍺 *${name}'s Pint — ${score}/10*`;

  const lines = [
    header,
    _verdict(score),
    '',
    `\`Head ratio  \` ${bar(bd.head_ratio ?? 5)}  ${bd.head_ratio ?? '—'}/10  _(${pct(bd.head_ratio_raw)} head)_`,
    `\`Texture     \` ${bar(bd.texture    ?? 5)}  ${bd.texture    ?? '—'}/10  _(${bd.bubble_count ?? '—'} bubbles)_`,
    `\`Colour sep  \` ${bar(bd.colour_sep ?? 5)}  ${bd.colour_sep ?? '—'}/10`,
    `\`Glass       \` ${bar(bd.glass_check ?? 5)}  ${bd.glass_check ?? '—'}/10  _(${bd.is_tulip ? 'tulip ✓' : 'wrong glass ✗'})_`,
    '',
    sg.comment ?? '💧 Send a mid-sip photo with *!splitg* to check the G split',
  ];

  if (warns.length) {
    lines.push('', `⚠️ _${warns.join(', ')}_`);
  }

  return lines.join('\n');
}

/**
 * Multi-glass reply — joins individual glass replies with a divider.
 */
function multiScoreReply(name, glasses) {
  if (glasses.length === 1) return scoreReply(name, glasses[0]);

  const parts = glasses.map((g, i) => scoreReply(name, g, i + 1));
  return parts.join('\n\n' + '─'.repeat(20) + '\n\n');
}

// ── Split-the-G reply ─────────────────────────────────────────

/**
 * Standalone !splitg result.
 * @param {object} splitg  - splitg sub-object from infer.py result
 * @param {string} name
 */
function splitgReply(splitg, name) {
  if (!splitg) {
    return `❓ Couldn't evaluate the G split, ${name}. Try a clearer mid-sip photo.`;
  }

  const lines = [
    `🔍 *Split the G — ${name}*`,
    '',
    splitg.comment ?? (splitg.detected ? '✅ G split detected' : '❌ G not split'),
  ];

  if (splitg.confidence != null) {
    lines.push(`Confidence: ${Math.round(splitg.confidence * 100)}%`);
  }

  if (splitg.status === 'not_evaluated') {
    lines.push('', '_Send a photo mid-sip, with the glass in frame, for best results._');
  }

  return lines.join('\n');
}

// ── Leaderboard ───────────────────────────────────────────────

/**
 * Current-month leaderboard.
 * @param {Array} rows  - from queries.leaderboard
 */
function leaderboard(rows) {
  if (!rows.length) {
    return '📊 No scores this month yet — be the first to submit a pint!';
  }

  const medals = ['🥇', '🥈', '🥉'];
  const lines  = rows.map((r, i) => {
    const pos   = medals[i] || `${i + 1}.`;
    const pints = r.submissions === 1 ? 'pint' : 'pints';
    return `${pos} *${r.user_name}*  ${r.best_score}/10  _(${r.submissions} ${pints}, avg ${r.avg_score})_`;
  });

  return ['🏆 *Guinness Leaderboard — This Month*', '', ...lines].join('\n');
}

// ── Monthly archive post ──────────────────────────────────────

/**
 * End-of-month summary posted by scheduler.js.
 * @param {string} month      - 'YYYY-MM'
 * @param {Array}  rows       - summarised rows for that month
 * @param {object} winner     - top row
 */
function monthlyArchive(month, rows, winner) {
  const [year, mon] = month.split('-');
  const monthName   = new Date(year, mon - 1).toLocaleString('en-GB', { month: 'long' });

  if (!rows.length) {
    return `📅 *${monthName} wrap-up* — No pints submitted this month. Shame.`;
  }

  const medals = ['🥇', '🥈', '🥉'];
  const podium = rows.slice(0, 3).map((r, i) =>
    `${medals[i]} *${r.user_name}*  ${r.best_score}/10  _(${r.submissions} pints)_`
  );

  const totalPints = rows.reduce((s, r) => s + r.submissions, 0);

  return [
    `📅 *${monthName} Leaderboard — Final Standings*`,
    '',
    ...podium,
    '',
    `${totalPints} pints scored across ${rows.length} drinkers.`,
    winner ? `\n🏆 *${winner.user_name}* wins ${monthName}. Respect.` : '',
    '',
    '_Scores reset. New month, fresh slate. 🍺_',
  ].filter(l => l !== null).join('\n');
}

// ── !me ───────────────────────────────────────────────────────

/**
 * Personal stats reply.
 * @param {string} name
 * @param {object} stats    - from queries.userStats
 * @param {object} rankRow  - from queries.userRank
 * @param {Array}  recent   - from queries.recentScores
 */
function me(name, stats, rankRow, recent) {
  if (!stats || stats.total === 0) {
    return `📊 No scores yet, ${name}! Attach a pint photo with *!score* to get started.`;
  }

  const recentLines = (recent ?? []).map((r, i) => {
    const bd = tryParse(r.breakdown) ?? {};
    const sg = tryParse(r.splitg)    ?? {};
    const parts = [`  ${i + 1}. *${r.pint_score}/10*`];
    if (bd.head_ratio != null) parts.push(`head ${bd.head_ratio}/10`);
    if (sg.comment)            parts.push(sg.comment);
    return parts.join(' · ');
  });

  return [
    `📊 *${name}'s Stats*`,
    '',
    `🏅 Rank:        #${rankRow?.rank ?? '—'}`,
    `🎯 Pints:       ${stats.total}`,
    `🏆 Best:        ${stars(stats.best)}  ${stats.best}/10`,
    `📈 Average:     ${stats.avg}/10`,
    `📉 Worst:       ${stats.worst}/10`,
    '',
    '_Recent:_',
    ...recentLines,
  ].join('\n');
}

// ── !help ─────────────────────────────────────────────────────

function help() {
  return [
    '🍺 *Guinness Bot — Commands*',
    '',
    '*Scoring*',
    '`!score` _(+ pint photo as caption)_',
    '  Scores head ratio, texture, colour separation & glass type.',
    '',
    '`!splitg` _(+ mid-sip photo as caption)_',
    '  Checks whether you\'ve correctly split the G.',
    '',
    '*Leaderboard*',
    '`!leaderboard` / `!lb`  — Monthly top 10',
    '`!me`  — Your personal stats & rank',
    '',
    '*Other*',
    '`!ping`  — Check the bot is alive',
    '`!help`  — This message',
    '',
    '_💡 Leaderboard resets at the start of each month._',
  ].join('\n');
}

// ── Verdict strings ───────────────────────────────────────────

function _verdict(score) {
  if (score >= 9.5) return '🏆 Perfection. Buy that barman a drink.';
  if (score >= 8.5) return '😤 Serious pint. Respect.';
  if (score >= 7.0) return '👍 Solid. No complaints.';
  if (score >= 5.5) return '😐 Drinkable. Just about.';
  if (score >= 4.0) return '😬 That\'s rough. Who poured this?';
  return                    '🚨 Criminal. Send it back.';
}

// ── Exports ───────────────────────────────────────────────────

const format = {
  scoreReply,
  multiScoreReply,
  splitgReply,
  leaderboard,
  monthlyArchive,
  me,
  help,
};

module.exports = { format };
