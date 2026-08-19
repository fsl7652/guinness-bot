/**
 * index.js
 *
 * WhatsApp client entry point.
 * Loads session, connects via Baileys, wires incoming messages to handler.js.
 * Starts the scheduler.
 */

const {
  default: makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  downloadMediaMessage,
  fetchLatestBaileysVersion,
} = require('@whiskeysockets/baileys');
const { Boom } = require('@hapi/boom');
const pino     = require('pino');
const path     = require('path');
const fs       = require('fs');

const { routeMessage } = require('./handler');
const { db }            = require('./db');
const scheduler         = require('./scheduler');

const logger      = pino({ level: process.env.LOG_LEVEL || 'info' });
const SESSION_DIR = path.join(__dirname, '..', 'session');

fs.mkdirSync(SESSION_DIR, { recursive: true });

// ── Connection ────────────────────────────────────────────────

async function connectToWhatsApp() {
  const { state, saveCreds }          = await useMultiFileAuthState(SESSION_DIR);
  const { version, isLatest }         = await fetchLatestBaileysVersion();
  logger.info({ version, isLatest }, '🔌 Baileys version');

  const sock = makeWASocket({
    version,
    auth:              state,
    printQRInTerminal: true,
    logger:            pino({ level: 'silent' }),
    browser:           ['Guinness Bot', 'Chrome', '1.0.0'],
    connectTimeoutMs:  30_000,
    retryRequestDelayMs: 2_000,
    // Keep message history minimal — saves RAM on Jetson
    getMessage: async () => undefined,
  });

  sock.ev.on('creds.update', saveCreds);

  sock.ev.on('connection.update', async ({ connection, lastDisconnect, qr }) => {
    if (qr) logger.info('📱 Scan the QR code to authenticate');

    if (connection === 'close') {
      const code = new Boom(lastDisconnect?.error)?.output?.statusCode;
      logger.warn({ code }, '🔴 Connection closed');

      if (code === DisconnectReason.loggedOut) {
        logger.error('🚪 Logged out — delete /session and restart to re-authenticate');
        process.exit(1);
      }

      const delay = code === DisconnectReason.restartRequired ? 1_000 : 5_000;
      logger.info({ delay }, '🔄 Reconnecting...');
      setTimeout(connectToWhatsApp, delay);
    }

    if (connection === 'open') {
      logger.info('✅ WhatsApp connected');
      scheduler.start(sock);
    }
  });

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;

    for (const msg of messages) {
      if (msg.key.fromMe)                                 continue;
      if (msg.key.remoteJid === 'status@broadcast')       continue;

      try {
        await routeMessage(sock, msg, { downloadMediaMessage });
      } catch (err) {
        logger.error({ err, msgId: msg.key.id }, '❌ Unhandled message error');
      }
    }
  });

  return sock;
}

// ── Graceful shutdown ─────────────────────────────────────────

function shutdown(signal) {
  logger.info({ signal }, '👋 Shutting down');
  scheduler.stop();
  db.close();
  process.exit(0);
}

process.on('SIGINT',  () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

process.on('uncaughtException',  (err) => logger.error({ err }, '💥 Uncaught exception'));
process.on('unhandledRejection', (err) => logger.error({ err }, '💥 Unhandled rejection'));

connectToWhatsApp();
