/**
 * scorer.js
 *
 * Calls infer.py via stdin/stdout JSON.
 * Handles three calling modes:
 *   1. Pint only         scoreImage(pintPath, null, name)
 *   2. Splitg only       scoreImage(null, splitgPath, name)
 *   3. Both              scoreImage(pintPath, splitgPath, name)
 *
 * infer.py protocol:
 *   stdin  → { pint_image?: "<base64>", splitg_image?: "<base64>", display_name?: string }
 *   stdout ← { glasses, message, pint_score, final, splitg }
 *           | { error, traceback? }
 */

'use strict';

const { spawn } = require('child_process');
const path      = require('path');
const fs        = require('fs');
const pino      = require('pino');

const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

const PYTHON_BIN    = process.env.PYTHON_BIN      || 'python3';
const INFER_SCRIPT  = path.join(__dirname, '..', 'ml', 'infer.py');
const INFER_TIMEOUT = parseInt(process.env.INFER_TIMEOUT_MS || '60000', 10);

/**
 * Score a pint image, a splitg image, or both.
 *
 * @param {string|null} pintPath    - absolute path to pint JPEG, or null
 * @param {string|null} splitgPath  - absolute path to mid-sip JPEG, or null
 * @param {string|null} displayName - sender's display name
 * @returns {Promise<object>}       infer.py output object
 */
async function scoreImage(pintPath, splitgPath = null, displayName = null) {
  if (!pintPath && !splitgPath) {
    throw new Error('scoreImage requires at least one image path');
  }

  const payload = {};

  if (pintPath) {
    payload.pint_image = fs.readFileSync(pintPath).toString('base64');
  }
  if (splitgPath && fs.existsSync(splitgPath)) {
    payload.splitg_image = fs.readFileSync(splitgPath).toString('base64');
  }
  if (displayName) {
    payload.display_name = displayName;
  }

  logger.info(
    { pintPath, splitgPath, displayName },
    '🤖 Calling infer.py'
  );

  const result = await runInference(payload);

  if (pintPath && (!result.glasses || result.glasses.length === 0)) {
    throw new Error('No glasses detected in image');
  }

  logger.info(
    { glasses: result.glasses?.length ?? 0, score: result.pint_score },
    '✅ Inference complete'
  );

  return result;
}

// ── Child process ─────────────────────────────────────────────

function runInference(payload) {
  return new Promise((resolve, reject) => {
    const proc = spawn(PYTHON_BIN, [INFER_SCRIPT], {
      env:  { ...process.env, PYTHONUNBUFFERED: '1' },
      cwd:  path.join(__dirname, '..', 'ml'),   // so relative imports work
    });

    let stdout = '';
    let stderr = '';

    proc.stdout.on('data', (chunk) => (stdout += chunk));
    proc.stderr.on('data', (chunk) => (stderr += chunk));

    const timer = setTimeout(() => {
      proc.kill('SIGKILL');
      reject(new Error(`infer.py timed out after ${INFER_TIMEOUT}ms`));
    }, INFER_TIMEOUT);

    proc.on('close', (code) => {
      clearTimeout(timer);

      if (stderr.trim()) {
        // Surface Python warnings / print statements from ML modules at debug level
        logger.debug({ stderr: stderr.slice(0, 800) }, 'infer.py stderr');
      }

      if (code !== 0) {
        logger.error({ code, stderr: stderr.slice(0, 400) }, '❌ infer.py non-zero exit');
        return reject(new Error(`infer.py exited ${code}: ${stderr.slice(0, 300)}`));
      }

      let parsed;
      try {
        parsed = JSON.parse(stdout.trim());
      } catch (err) {
        return reject(
          new Error(`Failed to parse infer.py output: ${err.message}\n${stdout.slice(0, 300)}`)
        );
      }

      if (parsed.error) {
        logger.error({ error: parsed.error, traceback: parsed.traceback }, '❌ Pipeline error');
        return reject(new Error(parsed.error));
      }

      resolve(parsed);
    });

    proc.on('error', (err) => {
      clearTimeout(timer);
      reject(new Error(`Failed to spawn infer.py: ${err.message}`));
    });

    proc.stdin.write(JSON.stringify(payload));
    proc.stdin.end();
  });
}

module.exports = { scoreImage };
