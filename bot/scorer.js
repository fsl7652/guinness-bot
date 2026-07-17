// bot/scorer.js - Updated with better media handling
const fs = require('fs');
const path = require('path');

// Stub mode configuration
const USE_STUB = true; // Set to false when using real ML model

async function processImage(media) {
    try {
        // Save the image for debugging (optional)
        const timestamp = Date.now();
        const filename = `data/raw/${timestamp}.jpg`;
        
        // Ensure directory exists
        const dir = path.dirname(filename);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        if (USE_STUB) {
            // Stub mode - return mock data
            console.log('Using stub scorer');
            
            // Save the image for later training data
            if (media.data) {
                const imageBuffer = Buffer.from(media.data, 'base64');
                fs.writeFileSync(filename, imageBuffer);
                console.log(`Saved image to ${filename}`);
            }
            
            // Return mock score
            return {
                pour: 7.5,
                splitg: { detected: true, confidence: 0.91 },
                final: 7.9
            };
        } else {
            // Real ML model - call Python script
            console.log('Using real ML model');
            
            // Save image temporarily
            if (media.data) {
                const imageBuffer = Buffer.from(media.data, 'base64');
                fs.writeFileSync(filename, imageBuffer);
                
                // Call Python script
                const { exec } = require('child_process');
                const pythonScript = path.join(__dirname, '../ml/infer.py');
                
                // Pass base64 image to Python
                const result = await new Promise((resolve, reject) => {
                    const pythonProcess = exec(`python ${pythonScript}`, {
                        maxBuffer: 1024 * 1024 * 10 // 10MB buffer
                    });
                    
                    pythonProcess.stdin.write(media.data);
                    pythonProcess.stdin.end();
                    
                    let output = '';
                    pythonProcess.stdout.on('data', (data) => {
                        output += data.toString();
                    });
                    
                    pythonProcess.on('close', (code) => {
                        if (code === 0) {
                            try {
                                const result = JSON.parse(output);
                                resolve(result);
                            } catch (e) {
                                reject(new Error('Invalid JSON from Python script'));
                            }
                        } else {
                            reject(new Error(`Python script exited with code ${code}`));
                        }
                    });
                });
                
                return result;
            }
        }
    } catch (error) {
        console.error('Image processing error:', error);
        throw error;
    }
}

module.exports = { processImage };