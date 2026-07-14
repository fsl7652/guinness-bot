const { Client, LocalAuth } = require('whatsapp-web.js');
const os = require('os');

function getChromePath() {
  if (os.platform() !== 'linux') return undefined; // use Puppeteer's bundled Chromium
  return '/usr/bin/chromium-browser'; // Jetson Nano / Linux server
}
const qrcode = require('qrcode-terminal');
const { handleMessage } = require('./handler');

const client = new Client({
  authStrategy: new LocalAuth({ dataPath: './session' }),
  puppeteer: {
    executablePath: getChromePath(),
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-dev-shm-usage'
    ]
  }
});

client.on('qr', qr => {
  console.log('Scan this QR code in WhatsApp:');
  qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
  console.log(`Bot ready — ${new Date().toISOString()}`);
});

client.on('auth_failure', () => {
  console.error('Auth failed — delete session/ folder and restart');
  process.exit(1);
});

client.on('disconnected', reason => {
  console.error('Disconnected:', reason);
  process.exit(1); // let pm2 restart it
});

client.on('message_create', msg => handleMessage(msg, client));

client.initialize();