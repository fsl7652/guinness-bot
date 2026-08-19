FROM node:18-slim

RUN apt-get update && apt-get install -y \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpangocairo-1.0-0 \
    libpango-1.0-0 \
    libcairo2 \
    libx11-6 \
    libx11-xcb1 \
    libxcb1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxrender1 \
    libxtst6 \
    fonts-liberation \
    wget \
    ca-certificates \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package*.json ./

# Let Puppeteer download its own bundled Chromium during npm install
ENV PUPPETEER_SKIP_CHROMIUM_DOWNLOAD=false
RUN npm install

COPY bot/ ./bot/

# Don't set CHROMIUM_PATH — let Puppeteer use its own bundled binary
ENV NODE_ENV=production

CMD ["node", "bot/index.js"]