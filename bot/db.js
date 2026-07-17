const Database = require('better-sqlite3');
const path     = require('path');

const db = new Database(path.join(__dirname, '../data/scores.db'));

db.exec(`
  CREATE TABLE IF NOT EXISTS scores (
    id             INTEGER PRIMARY KEY,
    user_jid       TEXT    NOT NULL,
    display_name   TEXT    NOT NULL,
    head_ratio     REAL,
    texture        REAL,
    colour_sep     REAL,
    glass_check    REAL,
    pint_score     REAL    NOT NULL,
    splitg_score   REAL,
    splitg_status  TEXT,
    image_file     TEXT,
    scored_at      DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE TABLE IF NOT EXISTS monthly_archive (
    id           INTEGER PRIMARY KEY,
    month        TEXT    NOT NULL,
    user_jid     TEXT    NOT NULL,
    display_name TEXT    NOT NULL,
    avg_score    REAL    NOT NULL,
    best_score   REAL    NOT NULL,
    pint_count   INTEGER NOT NULL,
    archived_at  DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE INDEX IF NOT EXISTS idx_user    ON scores(user_jid);
  CREATE INDEX IF NOT EXISTS idx_scored  ON scores(scored_at);
`);

// ── Writes ────────────────────────────────────────────────────

const insertScore = db.prepare(`
  INSERT INTO scores
    (user_jid, display_name, head_ratio, texture, colour_sep,
     glass_check, pint_score, splitg_score, splitg_status, image_file)
  VALUES
    (@user_jid, @display_name, @head_ratio, @texture, @colour_sep,
     @glass_check, @pint_score, @splitg_score, @splitg_status, @image_file)
`);

function saveScore(userJid, displayName, result, imageFile = null) {
  const b = result.breakdown || {};
  const s = result.splitg    || {};
  insertScore.run({
    user_jid:      userJid,
    display_name:  displayName,
    head_ratio:    b.head_ratio    ?? null,
    texture:       b.texture       ?? null,
    colour_sep:    b.colour_sep    ?? null,
    glass_check:   b.glass_check   ?? null,
    pint_score:    result.pint_score ?? result.final,
    splitg_score:  s.score         ?? null,
    splitg_status: s.status        ?? null,
    image_file:    imageFile,
  });
}

// ── Leaderboard queries ───────────────────────────────────────

const leaderboardQuery = db.prepare(`
  SELECT
    display_name,
    ROUND(AVG(pint_score), 1) AS avg,
    ROUND(MAX(pint_score), 1) AS best,
    COUNT(*)                  AS count
  FROM scores
  GROUP BY user_jid
  ORDER BY avg DESC
  LIMIT 10
`);

const monthlyLeaderboardQuery = db.prepare(`
  SELECT
    display_name,
    ROUND(AVG(pint_score), 1) AS avg,
    ROUND(MAX(pint_score), 1) AS best,
    COUNT(*)                  AS count
  FROM scores
  WHERE strftime('%Y-%m', scored_at) = ?
  GROUP BY user_jid
  ORDER BY avg DESC
  LIMIT 10
`);

const userStatsQuery = db.prepare(`
  SELECT
    ROUND(AVG(pint_score), 1) AS avg,
    ROUND(MAX(pint_score), 1) AS best,
    ROUND(MIN(pint_score), 1) AS worst,
    COUNT(*)                  AS count,
    SUM(CASE WHEN splitg_status = 'split' THEN 1 ELSE 0 END) AS splitg_count
  FROM scores
  WHERE user_jid = ?
`);

const recentScoresQuery = db.prepare(`
  SELECT pint_score, splitg_status, scored_at
  FROM scores
  WHERE user_jid = ?
  ORDER BY scored_at DESC
  LIMIT 5
`);

const archiveMonthQuery = db.prepare(`
  INSERT INTO monthly_archive
    (month, user_jid, display_name, avg_score, best_score, pint_count)
  SELECT
    strftime('%Y-%m', scored_at),
    user_jid,
    display_name,
    ROUND(AVG(pint_score), 1),
    ROUND(MAX(pint_score), 1),
    COUNT(*)
  FROM scores
  WHERE strftime('%Y-%m', scored_at) = ?
  GROUP BY user_jid
`);

// ── Reads ─────────────────────────────────────────────────────

function getLeaderboard() {
  return leaderboardQuery.all();
}

function getMonthlyLeaderboard(yearMonth) {
  return monthlyLeaderboardQuery.all(yearMonth);
}

function getUserStats(userJid) {
  const row = userStatsQuery.get(userJid);
  return row?.count ? row : null;
}

function getRecentScores(userJid) {
  return recentScoresQuery.all(userJid);
}

function archiveMonth(yearMonth) {
  return archiveMonthQuery.run(yearMonth);
}

function currentYearMonth() {
  return new Date().toISOString().slice(0, 7);
}

module.exports = {
  saveScore,
  getLeaderboard,
  getMonthlyLeaderboard,
  getUserStats,
  getRecentScores,
  archiveMonth,
  currentYearMonth,
};