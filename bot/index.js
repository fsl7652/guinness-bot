// bot/index.js - Simple working version for Linux
const { Client, LocalAuth } = require('whatsapp-web.js');
const qrcode = require('qrcode-terminal');
const fs = require('fs');
const path = require('path');

// Ensure directories exist
const dirs = ['data/raw', 'session'];
dirs.forEach(dir => {
    const fullPath = path.join(__dirname, '..', dir);
    if (!fs.existsSync(fullPath)) {
        fs.mkdirSync(fullPath, { recursive: true });
    }
});

console.log('🐧 Running on Linux');

// Find Chrome/Chromium
function findChrome() {
    const paths = [
        '/usr/bin/chromium-browser',
        '/usr/bin/chromium',
        '/usr/bin/google-chrome-stable',
        '/usr/bin/google-chrome'
    ];
    for (const p of paths) {
        if (fs.existsSync(p)) {
            console.log(`✅ Found Chrome at: ${p}`);
            return p;
        }
    }
    console.log('⚠️ Chrome not found, will use bundled Chromium');
    return undefined;
}

const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: path.join(__dirname, '..', 'session')
    }),
    puppeteer: {
        executablePath: findChrome(),
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu'
        ],
        headless: true
    }
});

client.on('qr', (qr) => {
    console.log('📱 Scan this QR code:');
    qrcode.generate(qr, { small: true });
});

client.on('ready', () => {
    console.log('✅ BOT IS READY!');
    console.log(`🕐 ${new Date().toISOString()}`);
    console.log('📱 Send !ping to test');
});

client.on('message', async (msg) => {
    if (msg.fromMe) return;
    
    console.log(`📩 Message: ${msg.body.substring(0, 50)}`);
    
    // Ping
    if (msg.body === '!ping') {
        await msg.reply('🏓 Pong!');
        return;
    }
    
    // Score with image
    if (msg.body.startsWith('!score')) {
        if (!msg.hasMedia) {
            await msg.reply('📸 Please send a photo with !score');
            return;
        }
        
        try {
            await msg.reply('📸 Processing your pint...');
            
            // This should work on Linux!
            const media = await msg.downloadMedia();
            
            if (media && media.data) {
                const timestamp = Date.now();
                const filename = path.join(__dirname, '..', 'data', 'raw', `${timestamp}.jpg`);
                const buffer = Buffer.from(media.data, 'base64');
                fs.writeFileSync(filename, buffer);
                
                await msg.reply(`🍺 Pint scored!\nPour: 7.5/10\nFinal Score: 7.9/10`);
                console.log(`✅ Image saved: ${filename}`);
            } else {
                await msg.reply('❌ Could not download image');
            }
        } catch (error) {
            console.error('❌ Error:', error.message);
            await msg.reply('❌ Error processing image');
        }
        return;
    }
    
    // Help
    if (msg.body === '!help' || msg.body === '!h') {
        await msg.reply(`📋 Commands:
!ping - Health check
!me - Your stats
!leaderboard - Top 10
!score [with photo] - Score a pint
!help - Show this menu`);
    }
});

client.on('disconnected', (reason) => {
    console.log('❌ Disconnected:', reason);
});

process.on('uncaughtException', (error) => {
    console.error('💥 Error:', error);
});

console.log('🚀 Starting bot...');
client.initialize().catch(error => {
    console.error('❌ Failed:', error.message);
});