// bot/handler.js - Using downloadMediaStream() from official example
const { MessageMedia } = require('whatsapp-web.js');
const fs = require('fs');
const path = require('path');

// Track processing to prevent duplicates
const processing = new Set();

async function handleMessage(msg, client) {
    const body = msg.body || '';
    const hasMedia = msg.hasMedia;
    const type = msg.type;
    
    console.log(`📩 Message from: ${msg.from}`);
    console.log(`📝 Body: ${body.substring(0, 50)}`);
    console.log(`📎 Has media: ${hasMedia}, Type: ${type}`);

    // ========== TEXT COMMANDS ==========
    
    if (body.toLowerCase() === '!ping') {
        await msg.reply('🏓 Pong!');
        return;
    }
    
    if (body.toLowerCase() === '!me') {
        await msg.reply('👤 Your stats: 0 points, 0 pints scored');
        return;
    }
    
    if (body.toLowerCase() === '!leaderboard') {
        await msg.reply('🏆 Top 10:\nNo scores yet!');
        return;
    }
    
    if (body.toLowerCase() === '!help' || body.toLowerCase() === '!h') {
        const helpText = `📋 Available commands:
!ping - Health check
!me - Your stats
!leaderboard - Top 10
!score [with photo] - Score a pint
!help - Show this menu`;
        await msg.reply(helpText);
        return;
    }

    // ========== SCORE COMMAND WITH IMAGE ==========
    
    if (body.toLowerCase().startsWith('!score')) {
        if (!hasMedia) {
            await msg.reply('📸 Please send a photo with !score');
            return;
        }
        
        if (type !== 'image') {
            await msg.reply('📸 Please send an image with !score');
            return;
        }
        
        const msgId = msg.id.id;
        if (processing.has(msgId)) {
            console.log('⏳ Message already being processed');
            return;
        }
        processing.add(msgId);
        
        try {
            console.log('📸 Processing image...');
            await msg.reply('📸 Processing your pint...');
            
            console.log('⏳ Downloading media using downloadMediaStream()...');
            
            // ===== USE downloadMediaStream() FROM OFFICIAL EXAMPLE =====
            const result = await msg.downloadMediaStream();
            
            if (!result) {
                console.error('❌ Media stream download returned undefined');
                await msg.reply('❌ Could not download your image. Please try again.');
                processing.delete(msgId);
                return;
            }
            
            console.log('✅ Media stream downloaded successfully!');
            console.log(`📊 MIME type: ${result.mimetype}`);
            console.log(`📊 File size: ${result.filesize} bytes`);
            console.log(`📊 Filename: ${result.filename || 'unknown'}`);
            
            // Save the image using the stream
            const timestamp = Date.now();
            const filename = path.join(__dirname, '..', 'data', 'raw', `${timestamp}.jpg`);
            
            // Ensure directory exists
            const dir = path.dirname(filename);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }
            
            // Write the stream to file
            const writeStream = fs.createWriteStream(filename);
            
            // Pipe the stream to file
            await new Promise((resolve, reject) => {
                result.stream.pipe(writeStream);
                writeStream.on('finish', resolve);
                writeStream.on('error', reject);
                result.stream.on('error', reject);
            });
            
            console.log(`✅ Image saved: ${filename}`);
            
            // Also get the base64 data for processing (convert the stream to base64)
            const fileBuffer = fs.readFileSync(filename);
            const base64Data = fileBuffer.toString('base64');
            
            // ====== SCORING LOGIC HERE ======
            // Now you can use base64Data for your ML model
            
            const mockScore = {
                pour: (6.5 + Math.random() * 3.0).toFixed(1),
                splitg: Math.random() > 0.3 ? '✅' : '❌',
                final: (7.0 + Math.random() * 2.5).toFixed(1)
            };
            
            const response = `🍺 Pint scored!
Pour: ${mockScore.pour}/10
Split G: ${mockScore.splitg}
Final Score: ${mockScore.final}/10`;
            
            await msg.reply(response);
            console.log('✅ Score processed successfully');
            
        } catch (error) {
            console.error('❌ Error processing score:', error);
            console.error('❌ Full error details:', error.stack);
            await msg.reply('❌ Error processing your pint. Please try again.');
        } finally {
            processing.delete(msgId);
        }
        return;
    }

    // ========== SEND MEDIA COMMAND ==========
    
    if (body.toLowerCase() === '!sendmedia' || body.toLowerCase() === '!sendpicture') {
        try {
            const imageFiles = fs.readdirSync(path.join(__dirname, '..', 'data', 'raw'));
            if (imageFiles.length > 0) {
                const latestImage = imageFiles[imageFiles.length - 1];
                const imagePath = path.join(__dirname, '..', 'data', 'raw', latestImage);
                const media = MessageMedia.fromFilePath(imagePath);
                await client.sendMessage(msg.from, media, { 
                    caption: '📸 Here\'s your latest scored pint!' 
                });
            } else {
                await msg.reply('No images found. Score a pint first!');
            }
        } catch (error) {
            console.error('Error sending media:', error);
            await msg.reply('❌ Error sending image');
        }
        return;
    }

    if (body.startsWith('!')) {
        await msg.reply('Unknown command. Send !help for available commands.');
    }
}

module.exports = { handleMessage };