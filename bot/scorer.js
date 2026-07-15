const { spawn } = require('child_process');
const path = require('path');

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