# Multi-User Relay Server Setup Guide

This guide explains how the multi-user collaborative training works and how to deploy your own relay server on Cloudflare Workers (free).

## How It Works

### The Problem
When multiple users want to train together, they need to exchange model weights. Direct peer-to-peer connections require:
- Being on the same local network, OR
- Port forwarding / firewall configuration
- Knowing each other's IP addresses

This is complicated for most users.

### The Solution: WebSocket Relay
A relay server acts as a middleman that routes messages between users:

```
┌─────────────┐                                         ┌─────────────┐
│   User A    │                                         │   User B    │
│  (Creator)  │                                         │  (Joiner)   │
└──────┬──────┘                                         └──────┬──────┘
       │                                                       │
       │  1. Create room                                       │
       │  ─────────────────►  ┌─────────────────────┐          │
       │                      │                     │          │
       │  2. Room code: ABC123│  Cloudflare Worker  │          │
       │  ◄─────────────────  │  (Relay Server)     │          │
       │                      │                     │          │
       │                      │  - Routes messages  │  3. Join ABC123
       │                      │  - No data storage  │  ◄───────┤
       │                      │  - Free forever     │          │
       │                      │                     │          │
       │  4. User B joined!   └─────────┬───────────┘          │
       │  ◄─────────────────────────────┼──────────────────────┤
       │                                │                      │
       │  5. Send weights ─────────────►│──────────────────────► Receive weights
       │                                │                      │
       │  6. Receive weights ◄──────────│◄───────────────────── Send weights
       │                                │                      │
```

### Why Cloudflare Workers?

| Feature | Cloudflare Workers | Other Options |
|---------|-------------------|---------------|
| Cost | **Free forever** | Usually paid |
| Credit card required | **No** | Usually yes |
| Setup time | **5 minutes** | Often complex |
| Concurrent connections | **100,000** | Often limited |
| Global availability | **Yes** (edge network) | Varies |
| Maintenance | **None** | Server management |

### Technical Details

**WebSocket Protocol:**
- Clients connect via `wss://` (secure WebSocket)
- Messages are JSON for control, binary for model weights
- Rooms are identified by 6-character codes (e.g., `ABC123`)

**Cloudflare Durable Objects:**
- Each room is managed by a Durable Object
- Handles WebSocket connections with hibernation (cost-efficient)
- State survives brief disconnections
- Automatically scales to handle load

**Message Flow:**
1. Creator sends `create_room` → Server responds with `room_created` + code
2. Joiner sends `join_room` with code → Server responds with `room_joined` + user list
3. Any message from a user is broadcast to all others in the room
4. Binary data (model weights) is forwarded without parsing

---

## Deploy Your Own Relay Server

### Prerequisites
- A Cloudflare account (free)
- Node.js installed on your computer
- 5 minutes of time

### Step 1: Create Cloudflare Account

1. Go to [cloudflare.com](https://cloudflare.com)
2. Click "Sign Up"
3. Enter email and password
4. Verify your email
5. **No credit card needed** for Workers free tier

### Step 2: Set Up Workers Subdomain

1. Log into the [Cloudflare Dashboard](https://dash.cloudflare.com)
2. Click "Workers & Pages" in the left sidebar
3. If prompted, create a workers.dev subdomain (e.g., `yourname.workers.dev`)
   - This is your free subdomain for hosting workers
   - Choose something memorable

### Step 3: Install Wrangler CLI

Open a terminal and run:

```bash
npm install -g wrangler
```

Or if you don't have npm:
```bash
# macOS with Homebrew
brew install node
npm install -g wrangler

# Or use the standalone installer from nodejs.org
```

### Step 4: Login to Cloudflare

```bash
wrangler login
```

This opens a browser window. Click "Allow" to authorize Wrangler.

### Step 5: Deploy the Relay Server

Navigate to the relay_server directory and deploy:

```bash
cd relay_server
npm install
wrangler deploy
```

You'll see output like:
```
Uploaded segmentation-relay (1.23 sec)
Deployed segmentation-relay triggers (0.42 sec)
  https://segmentation-relay.yourname.workers.dev
```

**Save this URL!** You'll need it in the next step.

### Step 6: Configure the App to Use Your Relay

1. Navigate to `segmentation_suite/network/`
2. Copy `relay_config.example.txt` to `relay_config.txt`
3. Edit `relay_config.txt` and add your relay URL:

```
wss://segmentation-relay.yoursubdomain.workers.dev/ws
```

That's it! The app reads this file automatically.

**Note:** `relay_config.txt` is gitignored so your personal URL won't be committed.

### Step 7: Test It!

1. Run the Segmentation Suite app
2. Go to the Setup page
3. Click "Create Session"
4. You should get a 6-character room code
5. On another computer, click "Join Session" and enter the code

---

## Troubleshooting

### "Room not found" error
- Room codes expire when the creator disconnects
- Make sure the creator's app is still running
- Codes are case-insensitive but stored as uppercase

### Connection drops after a few seconds
- Check your internet connection
- The relay automatically reconnects on brief disconnections
- If persistent, check Cloudflare status: [cloudflarestatus.com](https://www.cloudflarestatus.com)

### Deploy fails with "subdomain not found"
- Go to Cloudflare Dashboard → Workers & Pages
- Make sure you've created a workers.dev subdomain
- Wait a few minutes for DNS propagation

### "wrangler: command not found"
- Make sure Node.js is installed: `node --version`
- Reinstall wrangler: `npm install -g wrangler`
- Try using npx: `npx wrangler deploy`

### Checking Logs
To see real-time logs from your relay server:

```bash
cd relay_server
wrangler tail
```

This shows all connections, room creations, and any errors.

---

## Usage Limits (Free Tier)

Cloudflare Workers free tier includes:
- **100,000 requests per day** (plenty for most teams)
- **100,000 concurrent WebSocket connections**
- **10ms CPU time per request** (relay uses ~1ms)
- **Unlimited bandwidth**

For reference:
- A typical training session uses ~100 requests/hour
- You'd need 1,000 concurrent sessions to hit daily limits
- Model weight transfers (even large ones) count as single requests

---

## Security Notes

- **No data is stored**: The relay only forwards messages, nothing is saved
- **Room codes are random**: 6 characters from 32-character set = 1 billion combinations
- **Connections are encrypted**: All traffic uses WSS (WebSocket Secure)
- **Rooms auto-delete**: When the last user leaves, the room is gone

For additional security, you can:
1. Add authentication to the relay (modify `src/index.js`)
2. Use a custom domain with Cloudflare (hides workers.dev URL)
3. Enable Cloudflare Access for IP restrictions

---

## Updating the Relay Server

If updates are released, redeploy with:

```bash
cd relay_server
git pull  # if using git
wrangler deploy
```

Your room codes and active connections will be briefly interrupted during deploy.

---

## Architecture Reference

```
segmentation_suite/
├── network/
│   ├── client.py      # SyncClient - connects to relay or direct
│   ├── server.py      # AggregationServer - for LAN mode
│   └── protocol.py    # Message types and serialization
│
relay_server/
├── src/
│   └── index.js       # Cloudflare Worker code
├── wrangler.toml      # Cloudflare configuration
└── package.json       # Node.js dependencies
```

**Key Files:**
- `client.py:32` - `DEFAULT_RELAY_URL` setting
- `index.js` - Relay server logic (Room class handles WebSockets)
- `wrangler.toml` - Durable Object configuration

---

## FAQ

**Q: Can I use this for other projects?**
A: Yes! The relay is a generic WebSocket message router. Just connect and send JSON/binary messages.

**Q: What if Cloudflare changes their free tier?**
A: The relay is simple enough to run anywhere. You could adapt it for Deno Deploy, Fly.io, or any WebSocket-capable host.

**Q: Can multiple teams use the same relay?**
A: Yes, rooms are isolated by code. Different teams can use the same relay without seeing each other.

**Q: Is there a limit to room size?**
A: No hard limit. Tested with 10+ concurrent users. Performance depends on message frequency.

**Q: Can I see who's using my relay?**
A: Use `wrangler tail` to see real-time logs. No persistent analytics by default.
