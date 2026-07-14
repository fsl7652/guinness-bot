# Guinness Bot

## Setup

```bash
npm install
npm run dev       # stub mode, no model needed
npm start         # real mode, expects ml/infer.py
```

On first run, scan the QR code in WhatsApp on your burner number.
Session is saved to `session/` — back this up so you don't have to re-scan.

## Run permanently with pm2

```bash
npm install -g pm2
pm2 start bot/index.js --name guinness-bot
pm2 save
pm2 startup
```

## Commands

| Command | Description |
|---------|-------------|
| `!score` + photo | Score a pint |
| `!leaderboard` | Top 10 |
| `!me` | Your stats |
| `!ping` | Health check |

## Switching to real model

In `bot/scorer.js`, change:
```js
const USE_STUB = true;
```

The real scorer expects `ml/infer.py` to accept base64 image on stdin
and return JSON like:
```json
{ "pour": 7.5, "splitg": { "detected": true, "confidence": 0.91 }, "final": 7.9 }
```

## Project structure

```
bot/
  index.js      — WhatsApp client setup
  handler.js    — message routing, formatting
  scorer.js     — stub/real model switcher
  db.js         — SQLite leaderboard
data/
  raw/          — saved pint images (your training data)
  scores.db     — SQLite database
session/        — WhatsApp session (back this up)
ml/
  infer.py      — inference script (add later)
```