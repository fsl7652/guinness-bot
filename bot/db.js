/**
 * db.js
 *
 * SQLite schema and all prepared statements.
 *
 * Tables:
 *   scores          — every submitted pint score
 *   scores_archive  — monthly snapshot copied from scores before reset
 *   images          — training data registry (every saved image)
 *
 * The leaderboard resets monthly. At reset time, scheduler.js calls
 * archiveMonth() which copies the current month's data into scores_archive,
 * then truncates scores. All-time records are derived from scores_archive.
 */

const Database = require('better-sqlite3');
const path     = require('path');
const fs       = require('fs');

const DB_DIR  = path.join(__dirname, '..', 'data');
const DB_PATH = path.join(DB_DIR, 'guinness.db');

fs.mkdirSync(DB_DIR, { recursive: true });

const db = new Database(DB_PATH);
db.pragma('journal_mode = WAL');
db.pragma('foreign_keys = ON');

// ── Schema ────────────────────────────────────────────────────

db.exec(`
  -- Live scores (current month)
  CREATE TABLE IF NOT EXISTS scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    user_name   TEXT    NOT NULL,
    image_path  TEXT    NOT NULL,
    pint_score  REAL    NOT NULL,          -- 0–10, weighted aggregate
    breakdown   TEXT,                      -- JSON: {head_ratio, texture, colour_sep, glass_check}
    splitg      TEXT,                      -- JSON: {status, detected, confidence, comment}
    warnings    TEXT,                      -- JSON: string[]
    score_type  TEXT    DEFAULT 'pint',    -- 'pint' | 'splitg'
    month       TEXT    GENERATED ALWAYS AS (strftime('%Y-%m', scored_at)) VIRTUAL,
    scored_at   DATETIME DEFAULT CURRENT_TIMESTAMP
  );

  CREATE INDEX IF NOT EXISTS idx_scores_user   ON scores(user_id);
  CREATE INDEX IF NOT EXISTS idx_scores_score  ON scores(pint_score DESC);
  CREATE INDEX IF NOT EXISTS idx_scores_month  ON scores(month);

  -- Monthly archive — one row per user per month, best score
  CREATE TABLE IF NOT EXISTS scores_archive (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    month       TEXT    NOT NULL,           -- 'YYYY-MM'
    user_id     TEXT    NOT NULL,
    user_name   TEXT    NOT NULL,
    best_score  REAL    NOT NULL,
    avg_score   REAL    NOT NULL,
    submissions INTEGER NOT NULL,
    archived_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(month, user_id)
  );

  CREATE INDEX IF NOT EXISTS idx_archive_month ON scores_archive(month);
  CREATE INDEX IF NOT EXISTS idx_archive_score ON scores_archive(best_score DESC);

  -- Training image registry
  CREATE TABLE IF NOT EXISTS images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT    NOT NULL,
    filepath    TEXT    NOT NULL UNIQUE,
    image_type  TEXT    NOT NULL,           -- 'pint' | 'splitg'
    labelled    INTEGER DEFAULT 0,          -- 0 = unlabelled, 1 = in Label Studio
    label_data  TEXT,                       -- JSON from Label Studio export
    saved_at    TEXT    NOT NULL
  );

  CREATE INDEX IF NOT EXISTS idx_images_type     ON images(image_type);
  CREATE INDEX IF NOT EXISTS idx_images_labelled ON images(labelled);
`);

// ── Prepared statements ───────────────────────────────────────

const queries = {

  // ── Scores ──────────────────────────────────────────────────

  insertScore: db.prepare(`
    INSERT INTO scores
      (user_id, user_name, image_path, pint_score, breakdown, splitg, warnings, score_type)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
  `),

  lastScoreForUser: db.prepare(`
    SELECT id FROM scores
    WHERE user_id = ? AND score_type = 'pint'
    ORDER BY scored_at DESC
    LIMIT 1
  `),

  updateSplitg: db.prepare(`
    UPDATE scores SET splitg = ? WHERE id = ?
  `),

  // ── Leaderboard (current month) ──────────────────────────────

  leaderboard: db.prepare(`
    SELECT
      user_name,
      ROUND(MAX(pint_score), 1)  AS best_score,
      ROUND(AVG(pint_score), 1)  AS avg_score,
      COUNT(*)                   AS submissions
    FROM scores
    WHERE score_type = 'pint'
      AND month = strftime('%Y-%m', 'now')
    GROUP BY user_id
    ORDER BY best_score DESC, avg_score DESC
    LIMIT ?
  `),

  // ── User stats ───────────────────────────────────────────────

  userStats: db.prepare(`
    SELECT
      COUNT(*)                    AS total,
      ROUND(MAX(pint_score), 1)   AS best,
      ROUND(AVG(pint_score), 1)   AS avg,
      ROUND(MIN(pint_score), 1)   AS worst
    FROM scores
    WHERE user_id = ? AND score_type = 'pint'
  `),

  userRank: db.prepare(`
    SELECT rank FROM (
      SELECT user_id,
             RANK() OVER (ORDER BY MAX(pint_score) DESC) AS rank
      FROM scores
      WHERE score_type = 'pint'
        AND month = strftime('%Y-%m', 'now')
      GROUP BY user_id
    ) WHERE user_id = ?
  `),

  recentScores: db.prepare(`
    SELECT id, pint_score, breakdown, splitg, warnings, scored_at
    FROM scores
    WHERE user_id = ? AND score_type = 'pint'
    ORDER BY scored_at DESC
    LIMIT ?
  `),

  // ── Archive ──────────────────────────────────────────────────

  // Summarise current month per user for archiving
  summariseMonth: db.prepare(`
    SELECT
      strftime('%Y-%m', 'now')   AS month,
      user_id,
      user_name,
      MAX(pint_score)            AS best_score,
      AVG(pint_score)            AS avg_score,
      COUNT(*)                   AS submissions
    FROM scores
    WHERE score_type = 'pint'
      AND month = strftime('%Y-%m', 'now')
    GROUP BY user_id
  `),

  insertArchive: db.prepare(`
    INSERT OR REPLACE INTO scores_archive
      (month, user_id, user_name, best_score, avg_score, submissions)
    VALUES (?, ?, ?, ?, ?, ?)
  `),

  deleteCurrentMonth: db.prepare(`
    DELETE FROM scores
    WHERE month = strftime('%Y-%m', 'now')
  `),

  // All-time leaderboard from archive
  allTimeLeaderboard: db.prepare(`
    SELECT
      user_name,
      ROUND(MAX(best_score), 1)  AS all_time_best,
      ROUND(AVG(avg_score),  1)  AS overall_avg,
      SUM(submissions)           AS total_pints
    FROM scores_archive
    GROUP BY user_id
    ORDER BY all_time_best DESC
    LIMIT ?
  `),

  // Monthly winner (for the summary post)
  monthWinner: db.prepare(`
    SELECT user_name, best_score, submissions
    FROM scores_archive
    WHERE month = ?
    ORDER BY best_score DESC
    LIMIT 1
  `),

  // ── Images ───────────────────────────────────────────────────

  insertImage: db.prepare(`
    INSERT OR IGNORE INTO images (user_id, filepath, image_type, saved_at)
    VALUES (?, ?, ?, ?)
  `),

  unlabelledCount: db.prepare(`
    SELECT COUNT(*) AS count FROM images WHERE labelled = 0
  `),

  markLabelled: db.prepare(`
    UPDATE images SET labelled = 1, label_data = ? WHERE filepath = ?
  `),
};

// ── Archive helper (called by scheduler) ─────────────────────

function archiveMonth() {
  const rows = queries.summariseMonth.all();
  if (rows.length === 0) return 0;

  const insertMany = db.transaction((rows) => {
    for (const r of rows) {
      queries.insertArchive.run(
        r.month, r.user_id, r.user_name,
        r.best_score, r.avg_score, r.submissions
      );
    }
  });

  insertMany(rows);
  queries.deleteCurrentMonth.run();
  return rows.length;
}

module.exports = { db, queries, archiveMonth };
