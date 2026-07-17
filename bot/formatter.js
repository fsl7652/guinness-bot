/**
 * formatter.js
 * Converts infer.py result objects into WhatsApp message strings.
 * Keeping formatting here means handler.js stays thin.
 */

function formatScore(result, displayName = null) {
  return result.message || _buildMessage(result, displayName);
}

function _buildMessage(result, displayName) {
  const glasses = result.glasses ?? [result];
  const parts   = glasses.map((g, i) =>
    _formatGlass(g, glasses.length > 1 ? i + 1 : null, displayName)
  );
  return parts.join('\n\n' + '─'.repeat(20) + '\n\n');
}

function _formatGlass(g, index, displayName) {
  const b     = g.breakdown || {};
  const s     = g.splitg    || {};
  const score = g.pint_score ?? g.final;

  const header = [
    '🍺 *',
    displayName ? `${displayName}'s ` : '',
    index ? `Glass ${index} — ` : '',
    `${score}/10*`,
  ].join('');

  const lines = [
    header,
    g.verdict || '',
    '',
    `Head ratio:   ${b.head_ratio ?? '?'}/10  (${_pct(b.head_ratio_raw)} head)`,
    `Texture:      ${b.texture    ?? '?'}/10  (${b.bubble_count ?? '?'} bubbles)`,
    `Colour sep:   ${b.colour_sep ?? '?'}/10`,
    `Glass:        ${b.glass_check ?? '?'}/10  (${b.is_tulip ? 'tulip ✓' : 'wrong glass ✗'})`,
    '',
    s.comment || '💧 No split-the-G photo',
  ];

  if (g.warnings?.length) {
    lines.push('', `⚠️ _${g.warnings.join(', ')}_`);
  }

  return lines.filter(l => l !== undefined).join('\n');
}

function formatLeaderboard(rows, title = '🍺 *Guinness Leaderboard*') {
  if (!rows.length) return 'No scores yet — send a pint with !score';
  const medals = ['🥇', '🥈', '🥉'];
  const lines  = rows.map((r, i) =>
    `${medals[i] ?? `${i + 1}.`} *${r.display_name}* — ` +
    `${r.avg}/10 avg  ${r.best} best  ${r.count} pints`
  );
  return [title, '', ...lines].join('\n');
}

function formatMonthlyWinner(rows, month) {
  if (!rows.length) return null;
  const winner = rows[0];
  return [
    `🏆 *${month} Guinness Champion*`,
    '',
    `*${winner.display_name}* wins with ${winner.avg}/10 average`,
    `Best pint: ${winner.best}/10 over ${winner.count} attempts`,
    '',
    rows.length > 1
      ? `Runner up: ${rows[1].display_name} (${rows[1].avg}/10)`
      : '',
    '',
    '🍺 New month, new pints. Who takes it?',
  ].filter(Boolean).join('\n');
}

function formatUserStats(stats, recentScores, displayName) {
  if (!stats) return 'No scores yet — send a pint with !score';

  const trend = _trend(recentScores);
  const lines = [
    `📊 *${displayName}'s Stats*`,
    '',
    `Average:  ${stats.avg}/10`,
    `Best:     ${stats.best}/10`,
    `Worst:    ${stats.worst}/10`,
    `Pints:    ${stats.count}`,
    `G splits: ${stats.splitg_count ?? 0}`,
  ];

  if (trend) lines.push('', `Trend: ${trend}`);
  return lines.join('\n');
}

function _trend(recentScores) {
  if (!recentScores || recentScores.length < 3) return null;
  const scores = recentScores.map(r => r.pint_score);
  const first  = scores.slice(-3).reduce((a, b) => a + b) / 3;
  const last   = scores.slice(0, 3).reduce((a, b) => a + b) / 3;
  const diff   = last - first;
  if (diff >  0.5) return '📈 Improving';
  if (diff < -0.5) return '📉 Declining';
  return '➡️ Consistent';
}

function _pct(val) {
  if (val == null) return '?';
  return `${Math.round(val * 100)}%`;
}

module.exports = {
  formatScore,
  formatLeaderboard,
  formatMonthlyWinner,
  formatUserStats,
};