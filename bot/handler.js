const { processScore } = require('./scorer');
const { addScore, getLeaderboard, getUserScore } = require('./db');

async function handleMessage(msg, client) {
  try {
    if (msg.body === '!ping') {
      await msg.reply('pong');
      return;
    }

    if (msg.body === '!leaderboard') {
      const scores = getLeaderboard();
      const text = scores.map((s, i) => `${i + 1}. ${s.user}: ${s.score}`).join('\n');
      await msg.reply(text || 'No scores yet');
      return;
    }

    if (msg.body === '!me') {
      const score = getUserScore(msg.from);
      await msg.reply(`Your score: ${score || 'No scores yet'}`);
      return;
    }

    if (msg.hasMedia) {
      const media = await msg.downloadMedia();
      const result = await processScore(media.data, msg.from);
      addScore(msg.from, result.final);
      const verdict = getVerdict(result.final);
      await msg.reply(`Score: ${result.final}/10\n${verdict}`);
      return;
    }
  } catch (error) {
    console.error('Error:', error.message);
    try {
      await msg.reply('Error processing your message');
    } catch (e) {
      console.error('Failed to send error reply:', e.message);
    }
  }
}

function getVerdict(score) {
  if (score >= 9) return '🍻 Perfect pour!';
  if (score >= 7) return '✅ Great pint';
  if (score >= 5) return '⚠️ Decent';
  return '❌ Needs work';
}

module.exports = { handleMessage };