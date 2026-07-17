const { spawn } = require('child_process');
const path      = require('path');

const USE_STUB = process.env.USE_STUB !== 'false';

// ── Stub scorer ───────────────────────────────────────────────
function stubScore(glassCount = 1) {
  const glasses = Array.from({ length: glassCount }, (_, i) => {
    const headRatio  = +(Math.random() * 4 + 5).toFixed(1);
    const texture    = +(Math.random() * 4 + 5).toFixed(1);
    const colourSep  = +(Math.random() * 4 + 5).toFixed(1);
    const glassScore = Math.random() > 0.2 ? 10 : 4;
    const final      = +(
      headRatio * 0.375 +
      texture   * 0.3125 +
      colourSep * 0.25 +
      glassScore * 0.0625
    ).toFixed(1);
    const splitScore     = +(Math.random() * 10).toFixed(1);
    const splitDetected  = splitScore >= 5;

    return {
      index:      i + 1,
      pint_score: final,
      final,
      splitg: {
        status:     splitDetected ? 'split' : 'not_split',
        detected:   splitDetected,
        score:      splitScore,
        confidence: +(Math.random() * 0.4 + 0.6).toFixed(2),
        comment:    splitDetected ? '✅ Clean split.' : '❌ Nowhere near.',
      },
      verdict:   verdictText(final) + ' [stub]',
      breakdown: {
        head_ratio:      headRatio,
        texture,
        colour_sep:      colourSep,
        glass_check:     glassScore,
        head_ratio_raw:  +(Math.random() * 0.2 + 0.15).toFixed(2),
        bubble_count:    Math.floor(Math.random() * 60 + 20),
        is_tulip:        glassScore === 10,
      },
      warnings: ['scores are stubbed — model not loaded'],
    };
  });

  const message = glasses.map(g => formatStubMessage(g)).join('\n\n──────────────────\n\n');
  return {
    glasses,
    message,
    pint_score: glasses[0].final,
    final:      glasses[0].final,
    splitg:     glasses[0].splitg,
  };
}

function verdictText(score) {
  if (score >= 9.5) return '🏆 Perfection. Buy that barman a drink.';
  if (score >= 8.5) return '😤 Serious pint. Respect.';
  if (score >= 7.0) return '👍 Solid. No complaints.';
  if (score >= 5.5) return '😐 Drinkable. Just about.';
  if (score >= 4.0) return "😬 That's rough. Who poured this?";
  return                    '🚨 Criminal. Send it back.';
}

function formatStubMessage(g) {
  const b = g.breakdown;
  return [
    `🍺 *Glass ${g.index} — ${g.final}/10*`,
    g.verdict,
    '',
    `Head ratio:   ${b.head_ratio}/10  (${Math.round(b.head_ratio_raw * 100)}% head)`,
    `Texture:      ${b.texture}/10  (${b.bubble_count} bubbles)`,
    `Colour sep:   ${b.colour_sep}/10`,
    `Glass:        ${b.glass_check}/10  (${b.is_tulip ? 'tulip ✓' : 'wrong glass ✗'})`,
    '',
    g.splitg.comment,
    '',
    `⚠️ _${g.warnings[0]}_`,
  ].join('\n');
}

// ── Real scorer ───────────────────────────────────────────────
function realScore(pintBase64, splitgBase64 = null, displayName = null) {
  return new Promise((resolve, reject) => {
    const py = spawn('python3', [path.join(__dirname, '../ml/infer.py')]);
    let stdout = '';
    let stderr = '';

    py.stdout.on('data', d => stdout += d);
    py.stderr.on('data', d => stderr += d);

    py.on('close', code => {
      if (stderr) console.error('[infer.py]', stderr);
      if (code !== 0) return reject(new Error(`infer.py exited ${code}: ${stderr}`));
      try {
        const result = JSON.parse(stdout);
        if (result.error) return reject(new Error(result.error));
        resolve(result);
      } catch {
        reject(new Error(`Bad JSON from infer.py: ${stdout.slice(0, 200)}`));
      }
    });

    py.on('error', reject);

    const payload = { pint_image: pintBase64 };
    if (splitgBase64) payload.splitg_image = splitgBase64;
    if (displayName)  payload.display_name = displayName;

    py.stdin.write(JSON.stringify(payload));
    py.stdin.end();
  });
}

async function scoreImage(pintBase64, splitgBase64 = null, displayName = null) {
  if (USE_STUB) return stubScore();
  return realScore(pintBase64, splitgBase64, displayName);
}

module.exports = { scoreImage };