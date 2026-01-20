/**
 * Segmentation Suite - WebSocket Relay Server for Cloudflare Workers
 *
 * Uses Durable Objects for room management and WebSocket handling.
 * Deploy with: wrangler deploy
 */

// Generate a random room code (6 uppercase alphanumeric characters)
function generateRoomCode() {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Removed confusing chars: I, O, 0, 1
  let code = '';
  for (let i = 0; i < 6; i++) {
    code += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return code;
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Health check endpoint
    if (url.pathname === '/' || url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'ok',
        service: 'Segmentation Suite Relay',
        version: '1.0.0'
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // WebSocket upgrade for room connections
    // Supports: /ws or /room/<code> for direct room join
    if (url.pathname === '/ws' || url.pathname.startsWith('/room/')) {
      // Check for WebSocket upgrade
      const upgradeHeader = request.headers.get('Upgrade');
      if (!upgradeHeader || upgradeHeader !== 'websocket') {
        return new Response('Expected WebSocket', { status: 426 });
      }

      // Extract room code from URL if provided (for backwards compatibility)
      let roomCode = null;
      if (url.pathname.startsWith('/room/')) {
        roomCode = url.pathname.split('/room/')[1]?.toUpperCase();
      }

      // Get or create a Durable Object for this connection
      // We use a lobby DO to handle room creation/joining
      const lobbyId = env.ROOMS.idFromName('lobby');
      const lobby = env.ROOMS.get(lobbyId);

      // Forward the WebSocket request to the lobby
      return lobby.fetch(request);
    }

    // REST API for room creation (alternative to WebSocket-based creation)
    if (url.pathname === '/api/create-room' && request.method === 'POST') {
      const roomCode = generateRoomCode();
      return new Response(JSON.stringify({ room_code: roomCode }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    return new Response('Not Found', { status: 404 });
  }
};

/**
 * Durable Object for managing WebSocket rooms
 *
 * This handles:
 * - Room creation and joining
 * - Message routing between clients in the same room
 * - User presence tracking
 * - Automatic cleanup when rooms empty
 */
export class Room {
  constructor(state, env) {
    this.state = state;
    this.env = env;

    // Map of room code -> Set of WebSocket connections
    this.rooms = new Map();

    // Map of WebSocket -> client info { roomCode, userId, displayName }
    this.clients = new Map();

    // Restore state from hibernation
    this.state.getWebSockets().forEach(ws => {
      const attachment = ws.deserializeAttachment();
      if (attachment) {
        this.clients.set(ws, attachment);
        if (attachment.roomCode) {
          if (!this.rooms.has(attachment.roomCode)) {
            this.rooms.set(attachment.roomCode, new Set());
          }
          this.rooms.get(attachment.roomCode).add(ws);
        }
      }
    });
  }

  async fetch(request) {
    // Handle WebSocket upgrade
    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);

    // Accept the WebSocket connection with hibernation support
    this.state.acceptWebSocket(server);

    // Store initial client state (will be updated on create_room/join_room)
    const initialState = { roomCode: null, userId: null, displayName: null };
    this.clients.set(server, initialState);
    server.serializeAttachment(initialState);

    return new Response(null, { status: 101, webSocket: client });
  }

  async webSocketMessage(ws, message) {
    try {
      // Handle binary data (model weights) - forward to room
      if (message instanceof ArrayBuffer) {
        const clientInfo = this.clients.get(ws);
        if (clientInfo?.roomCode) {
          this.broadcastToRoom(clientInfo.roomCode, message, ws);
        }
        return;
      }

      // Parse JSON message
      const data = JSON.parse(message);
      const { type, payload } = data;

      switch (type) {
        case 'create_room':
          await this.handleCreateRoom(ws, payload);
          break;

        case 'join_room':
          await this.handleJoinRoom(ws, payload);
          break;

        case 'leave_room':
          await this.handleLeaveRoom(ws);
          break;

        default:
          // Forward all other messages to room members
          await this.handleForwardMessage(ws, data);
          break;
      }
    } catch (err) {
      console.error('Error handling message:', err);
      this.sendToClient(ws, {
        type: 'error',
        payload: { error: 'Invalid message format', details: err.message }
      });
    }
  }

  async webSocketClose(ws, code, reason) {
    await this.handleClientDisconnect(ws);
  }

  async webSocketError(ws, error) {
    console.error('WebSocket error:', error);
    await this.handleClientDisconnect(ws);
  }

  // Generate room code
  generateRoomCode() {
    const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    let code = '';
    for (let i = 0; i < 6; i++) {
      code += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return code;
  }

  // Create a new room
  async handleCreateRoom(ws, payload) {
    const { user_id, display_name } = payload;

    // Generate unique room code
    let roomCode;
    do {
      roomCode = this.generateRoomCode();
    } while (this.rooms.has(roomCode));

    // Create room and add creator
    this.rooms.set(roomCode, new Set([ws]));
    const clientInfo = { roomCode, userId: user_id, displayName: display_name };
    this.clients.set(ws, clientInfo);

    // Persist state for hibernation recovery
    ws.serializeAttachment(clientInfo);

    // Send confirmation
    this.sendToClient(ws, {
      type: 'room_created',
      payload: { room_code: roomCode }
    });

    console.log(`Room ${roomCode} created by ${display_name}`);
  }

  // Join existing room
  async handleJoinRoom(ws, payload) {
    const { room_code, user_id, display_name } = payload;
    const roomCode = room_code.toUpperCase();

    // Check if room exists
    if (!this.rooms.has(roomCode)) {
      this.sendToClient(ws, {
        type: 'error',
        payload: { error: 'Room not found', code: 'ROOM_NOT_FOUND' }
      });
      return;
    }

    // Add client to room
    const room = this.rooms.get(roomCode);
    room.add(ws);
    const clientInfo = { roomCode, userId: user_id, displayName: display_name };
    this.clients.set(ws, clientInfo);

    // Persist state for hibernation recovery
    ws.serializeAttachment(clientInfo);

    // Get user list
    const users = this.getRoomUsers(roomCode);

    // Send confirmation to joiner
    this.sendToClient(ws, {
      type: 'room_joined',
      payload: { room_code: roomCode, users }
    });

    // Notify others in room
    this.broadcastToRoom(roomCode, JSON.stringify({
      type: 'user_joined',
      payload: { user_id, display_name, users }
    }), ws);

    console.log(`${display_name} joined room ${roomCode}`);
  }

  // Leave room
  async handleLeaveRoom(ws) {
    await this.handleClientDisconnect(ws);
  }

  // Forward message to room
  async handleForwardMessage(ws, data) {
    const clientInfo = this.clients.get(ws);
    if (!clientInfo?.roomCode) return;

    // Add sender info
    data.sender_id = clientInfo.userId;
    data.sender_name = clientInfo.displayName;

    this.broadcastToRoom(clientInfo.roomCode, JSON.stringify(data), ws);
  }

  // Handle client disconnect
  async handleClientDisconnect(ws) {
    const clientInfo = this.clients.get(ws);
    if (!clientInfo?.roomCode) {
      this.clients.delete(ws);
      return;
    }

    const { roomCode, userId, displayName } = clientInfo;
    const room = this.rooms.get(roomCode);

    if (room) {
      room.delete(ws);

      // Notify others
      const users = this.getRoomUsers(roomCode);
      this.broadcastToRoom(roomCode, JSON.stringify({
        type: 'user_left',
        payload: { user_id: userId, display_name: displayName, users }
      }));

      // Clean up empty rooms
      if (room.size === 0) {
        this.rooms.delete(roomCode);
        console.log(`Room ${roomCode} deleted (empty)`);
      }
    }

    this.clients.delete(ws);
    console.log(`${displayName} disconnected from room ${roomCode}`);
  }

  // Get list of users in a room
  getRoomUsers(roomCode) {
    const room = this.rooms.get(roomCode);
    if (!room) return [];

    const users = [];
    for (const ws of room) {
      const info = this.clients.get(ws);
      if (info) {
        users.push({
          user_id: info.userId,
          display_name: info.displayName
        });
      }
    }
    return users;
  }

  // Send to single client
  sendToClient(ws, data) {
    try {
      const message = typeof data === 'string' ? data : JSON.stringify(data);
      ws.send(message);
    } catch (err) {
      console.error('Error sending to client:', err);
    }
  }

  // Broadcast to all clients in room except sender
  broadcastToRoom(roomCode, message, excludeWs = null) {
    const room = this.rooms.get(roomCode);
    if (!room) return;

    const data = typeof message === 'string' ? message : message;

    for (const ws of room) {
      if (ws !== excludeWs) {
        try {
          ws.send(data);
        } catch (err) {
          console.error('Error broadcasting to client:', err);
        }
      }
    }
  }
}
