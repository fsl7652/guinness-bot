const { spawn } = require('child_process');
const path = require('path');

const USE_STUB = process.env.USE_STUB === 'true' || true;

function stubScore() {
  const pour = +(Math.random() * 4 + 6).toFixed(1);
  const splitDetected = Math.random() > 0.5;
  const splitConfidence = +(Math.random() * 0.4 + 0.6).toFixed(2);
  const splitBonus = splitDetected ? 1.5 : 0;
  const final = +Math.min(10, pour * 0.85 + splitBonus).toFixed(1);

  return { pour, splitg: { detected: splitDetected, confidence: splitConfidence }, final };
}

function realScore(imageBase64) {
  return new Promise((resolve, reject) => {
    const py = spawn('python3', [path.join(__dirname, '../ml/infer.py')]);
    let stdout = '';
    let stderr = '';

    py.stdout.on('data', d => stdout += d);
    py.stderr.on('data', d => stderr += d);

    py.on('close', code => {
      if (code !== 0) return reject(new Error(`infer.py exited ${code}: ${stderr}`));
      try {
        resolve(JSON.parse(stdout));
      } catch {
        reject(new Error(`Bad JSON from infer.py: ${stdout}`));
      }
    });

    py.on('error', reject);
    py.stdin.write(imageBase64);
    py.stdin.end();
  });
}

async function scoreImage(imageBase64) {
  if (USE_STUB) return stubScore();
  return realScore(imageBase64);
}

module.exports = { scoreImage };