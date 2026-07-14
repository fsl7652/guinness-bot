const Database = require('better-sqlite3');
const path = require('path');

const db = new Database(path.join(__dirname, '../data/scores.db'));

db.exec(`
  CREATE TABLE IF NOT EXISTS scores (
    id           INTEGER PRIMARY KEY,
    user_jid     TEXT    NOT NULL,
    display_name TEXT    NOT NULL,
    pour_score   REAL    NOT NULL,
    splitg       INTEGER NOT NULL,
    final_score  REAL    NOT NULL,
    image_file   TEXT,
    scored_at    DATETIME DEFAULT CURRENT_TIMESTAMP
  );
  CREATE INDEX IF NOT EXISTS idx_user ON scores(user_jid);
`);

const insertScore = db.prepare(`
  INSERT INTO scores (user_jid, display_name, pour_score, splitg, final_score, image_file)
  VALUES (@user_jid, @display_name, @pour_score, @splitg, @final_score, @image_file)
`);

const leaderboardQuery = db.prepare(`
  SELECT
    display_name,
    ROUND(AVG(final_score), 1) as avg,
    ROUND(MAX(final_score), 1) as best,
    COUNT(*)                   as count
  FROM scores
  GROUP BY user_jid
  ORDER BY avg DESC
  LIMIT 10
`);

const userStatsQuery = db.prepare(`
  SELECT
    ROUND(AVG(final_score), 1) as avg,
    ROUND(MAX(final_score), 1) as best,
    COUNT(*)                   as count
  FROM scores
  WHERE user_jid = ?
`);

function saveScore(userJid, displayName, result, imageFile) {
  insertScore.run({
    user_jid:     userJid,
    display_name: displayName,
    pour_score:   result.pour,
    splitg:       result.splitg.detected ? 1 : 0,
    final_score:  result.final,
    image_file:   imageFile
  });
}

function getLeaderboard() {
  return leaderboardQuery.all();
}

function getUserStats(userJid) {
  const row = userStatsQuery.get(userJid);
  return row?.count ? row : null;
}

module.exports = { saveScore, getLeaderboard, getUserStats };