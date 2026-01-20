# Segmentation Suite Relay Server

WebSocket relay server for multi-user collaborative training, deployed on Cloudflare Workers.

**For detailed setup instructions, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

## Quick Start

```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare (free account)
wrangler login

# Deploy
cd relay_server
npm install
wrangler deploy
```

Then update `segmentation_suite/network/client.py` line 32 with your URL.

## Features

- **Free forever** - Cloudflare Workers free tier (no credit card required)
- **100K concurrent connections** - More than enough for any team
- **Global edge network** - Fast connections worldwide
- **Hibernatable WebSockets** - Efficient resource usage
- **Auto-scaling** - Handles any load automatically

## Usage

### Creating a Room (Host)
```python
from segmentation_suite.network import SyncClient

client = SyncClient()
client.room_created.connect(lambda code: print(f"Share this code: {code}"))
client.create_relay_room("HostName")
```

### Joining a Room (Others)
```python
client = SyncClient()
client.connect_relay("ABC123", "UserName")
```

## How It Works

```
┌─────────────┐         ┌─────────────────────────┐         ┌─────────────┐
│   User A    │◄───────►│  Cloudflare Workers     │◄───────►│   User B    │
│  (room:X)   │   WSS   │  (Durable Object/Room)  │   WSS   │  (room:X)   │
└─────────────┘         └─────────────────────────┘         └─────────────┘
```

1. User A creates a room → Gets code like `ABC123`
2. User A shares code with User B
3. User B joins with code `ABC123`
4. Both users can now send/receive model weights
5. Messages are routed only to users in the same room

## Message Protocol

### Client → Server

```json
{"type": "create_room", "payload": {"user_id": "xxx", "display_name": "Alice"}}
{"type": "join_room", "payload": {"room_code": "ABC123", "user_id": "xxx", "display_name": "Bob"}}
{"type": "leave_room", "payload": {}}
```

### Server → Client

```json
{"type": "room_created", "payload": {"room_code": "ABC123"}}
{"type": "room_joined", "payload": {"room_code": "ABC123", "users": [...]}}
{"type": "user_joined", "payload": {"user_id": "xxx", "display_name": "Bob", "users": [...]}}
{"type": "user_left", "payload": {"user_id": "xxx", "display_name": "Bob", "users": [...]}}
{"type": "error", "payload": {"error": "Room not found", "code": "ROOM_NOT_FOUND"}}
```

### Forwarded Messages

All other message types (like `weights_push`, `global_model`) are forwarded to all other clients in the room. Binary data (model weights) is also forwarded.

## Local Development

```bash
cd relay_server
npm install
wrangler dev
```

This starts a local server at `http://localhost:8787`. 
Connect with: `ws://localhost:8787/ws`

## Monitoring

View live logs:
```bash
wrangler tail
```

## Limits (Free Tier)

- 100,000 requests/day
- 100,000 concurrent WebSocket connections
- 10ms CPU time per request (plenty for relay)
- Unlimited bandwidth

For most teams, you'll never hit these limits.

## Troubleshooting

### "Room not found" error
- Room codes are case-insensitive but stored uppercase
- Rooms are deleted when the last user leaves
- Check that the room creator is still connected

### Connection drops
- Cloudflare keeps WebSockets alive for up to 30 days
- Implement reconnection logic in your client (already done in SyncClient)

### Deploy fails
- Make sure you're logged in: `wrangler login`
- Check wrangler.toml is in the current directory
