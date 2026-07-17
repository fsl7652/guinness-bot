// bot/index.js - Simple WSL version that actually works
const { Client, LocalAuth, MessageMedia } = require('whatsapp-web.js');
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

console.log('🐧 Running on Linux (WSL)');

// Client with WSL-optimized settings
const client = new Client({
    authStrategy: new LocalAuth({
        dataPath: path.join(__dirname, '..', 'session')
    }),
    puppeteer: {
        executablePath: '/usr/bin/chromium-browser',
        args: [
            '--no-sandbox',
            '--disable-setuid-sandbox',
            '--disable-dev-shm-usage',
            '--disable-gpu'
        ],
        headless: true
    }
});

// QR Code
client.on('qr', (qr) => {
    console.log('📱 Scan this QR code:');
    qrcode.generate(qr, { small: true });
});

// Ready
client.on('ready', () => {
    console.log('✅ BOT IS READY!');
    console.log(`🕐 ${new Date().toISOString()}`);
});

// Message handler
client.on('message', async (msg) => {
    if (msg.fromMe) return;
    
    console.log(`📩 Message: ${msg.body}`);
    
    // Simple ping
    if (msg.body === '!ping') {
        await msg.reply('🏓 Pong!');
        return;
    }
    
    // Score with image
    if (msg.body.startsWith('!score') && msg.hasMedia) {
        try {
            await msg.reply('📸 Processing your pint...');
            
            // This actually works on Linux!
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
            console.error('Error:', error);
            await msg.reply('❌ Error processing image');
        }
        return;
    }
    
    // Help
    if (msg.body === '!help') {
        await msg.reply(`Commands:\n!ping\n!score [with photo]\n!help`);
    }
});

// Start
console.log('🚀 Starting bot...');
client.initialize().catch(console.error);