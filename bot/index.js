const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const os     = require('os');
const fs     = require('fs');

const { handleMessage }  = require('./handler');
const { startScheduler } = require('./scheduler');

const CHROME_PATH = process.env.CHROME_EXECUTABLE_PATH || (() => {
  const bundled = '/app/node_modules/puppeteer-core/.local-chromium/linux-1045629/chrome-linux/chrome';
  const exists  = fs.existsSync(bundled);
  console.log(`[chrome] bundled path exists: ${exists} → ${bundled}`);
  if (exists) return bundled;
  if (os.platform() !== 'linux') return undefined;
  return '/usr/bin/chromium-browser';
})();

console.log(`[chrome] using: ${CHROME_PATH}`);

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: '/app/session' }),
  puppeteer: {
    executablePath: CHROME_PATH,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage',
      '--disable-gpu',
      '--disable-software-rasterizer',
      '--disable-extensions',
      '--no-first-run',
      '--no-zygote',
      '--single-process',
    ]
  }
});

client.on('qr', qr => {
  console.log('Scan this QR code in WhatsApp:');
  qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
  console.log(`Bot ready — ${new Date().toISOString()}`);
  startScheduler(client, process.env.GROUP_JID);
});

client.on('auth_failure', () => {
  console.error('Auth failed — delete session/ folder and restart');
  process.exit(1);
});

client.on('disconnected', reason => {
  console.error('Disconnected:', reason);
  process.exit(1);
});

client.on('message', msg => {
  console.log('MSG body:', msg.body, '| hasMedia:', msg.hasMedia, '| type:', msg.type);
  handleMessage(msg, client);
});

client.initialize();