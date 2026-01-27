# Redis Setup Guide for Sarah Voice Agent

This guide covers setting up Redis for persistent memory storage, enabling the agent to remember returning users across sessions.

## Why Redis?

Redis provides:
- **Fast reads/writes** - Sub-millisecond latency for memory lookups
- **Data persistence** - Survives server restarts
- **TTL support** - Automatic expiration of old sessions
- **Cloud-ready** - Easy deployment with managed services

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Memory Flow                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Incoming Call                                              │
│        │                                                     │
│        ▼                                                     │
│   ┌─────────────────┐                                       │
│   │ Extract Phone # │                                       │
│   └────────┬────────┘                                       │
│            │                                                 │
│            ▼                                                 │
│   ┌─────────────────┐     ┌─────────────────┐              │
│   │  Hash Phone #   │────▶│  Redis Lookup   │              │
│   └─────────────────┘     │  phone:{hash}   │              │
│                           └────────┬────────┘              │
│                                    │                        │
│            ┌───────────────────────┼───────────────────┐   │
│            │                       │                   │   │
│            ▼                       ▼                   │   │
│   ┌─────────────────┐     ┌─────────────────┐         │   │
│   │  New User       │     │ Returning User  │         │   │
│   │  Create ID      │     │ Load Profile    │         │   │
│   └─────────────────┘     └─────────────────┘         │   │
│            │                       │                   │   │
│            └───────────┬───────────┘                   │   │
│                        │                               │   │
│                        ▼                               │   │
│           ┌─────────────────────────┐                  │   │
│           │   Personalized Greeting  │                  │   │
│           │   + Context from Memory  │                  │   │
│           └─────────────────────────┘                  │   │
│                                                         │   │
└─────────────────────────────────────────────────────────────┘
```

## Redis Key Structure

| Key Pattern | Data | TTL |
|-------------|------|-----|
| `sarah:phone:{hash}` | User ID | 90 days |
| `sarah:user:{id}:profile` | Preferences, last summary | 90 days |
| `sarah:user:{id}:sessions` | List of session IDs | 90 days |
| `sarah:session:{id}` | Full session data | 24 hours |

---

## Option 1: Upstash (Recommended for Serverless)

**Best for:** Low-to-medium traffic, pay-per-request, global edge

### Setup Steps:

1. **Create account** at [upstash.com](https://upstash.com)

2. **Create database:**
   - Click "Create Database"
   - Select region closest to your server
   - Choose "Regional" (cheaper) or "Global" (faster worldwide)

3. **Get connection string:**
   - Go to database details
   - Copy the "REST URL" and "Token"
   - Or use the Redis URL for traditional connection

4. **Configure .env:**
   ```env
   REDIS_URL=redis://default:xxxxx@eu1-fitting-asp-12345.upstash.io:6379
   ```

### Pricing:
- Free: 10K commands/day
- Pay-as-you-go: $0.2 per 100K commands

---

## Option 2: Redis Cloud (Redis Labs)

**Best for:** Production workloads, enterprise features

### Setup Steps:

1. **Create account** at [redis.com/try-free](https://redis.com/try-free/)

2. **Create subscription:**
   - Choose "Essentials" for free tier (30MB)
   - Select cloud provider (AWS/GCP/Azure)
   - Pick region

3. **Create database:**
   - Default settings work fine
   - Enable "Redis Cloud Essentials"

4. **Get credentials:**
   - Copy the public endpoint
   - Get password from "Security" tab

5. **Configure .env:**
   ```env
   REDIS_HOST=redis-12345.c123.us-east-1-2.ec2.cloud.redislabs.com
   REDIS_PORT=12345
   REDIS_PASSWORD=your-password-here
   REDIS_SSL=true
   ```

### Pricing:
- Free: 30MB storage
- Essentials: From $7/month

---

## Option 3: AWS ElastiCache

**Best for:** AWS infrastructure, high availability

### Setup Steps:

1. **Go to ElastiCache** in AWS Console

2. **Create cluster:**
   - Engine: Redis
   - Node type: cache.t3.micro (free tier eligible)
   - Number of replicas: 0 (for dev)

3. **Configure security group:**
   - Allow inbound TCP 6379 from your app's security group

4. **Get endpoint:**
   - Copy the Primary Endpoint

5. **Configure .env:**
   ```env
   REDIS_HOST=your-cluster.abc123.0001.use1.cache.amazonaws.com
   REDIS_PORT=6379
   REDIS_SSL=true
   ```

### Pricing:
- Free tier: 750 hours/month of cache.t3.micro
- After: ~$12/month for smallest instance

---

## Option 4: Local Redis (Development)

### Windows (WSL2):

```bash
# Install Redis in WSL
sudo apt update
sudo apt install redis-server

# Start Redis
sudo service redis-server start

# Test connection
redis-cli ping
# Should return: PONG
```

### Docker:

```bash
# Run Redis container
docker run -d --name sarah-redis -p 6379:6379 redis:alpine

# With persistence
docker run -d --name sarah-redis -p 6379:6379 \
  -v redis-data:/data redis:alpine redis-server --appendonly yes
```

### Windows Native:
Download from: [github.com/microsoftarchive/redis/releases](https://github.com/microsoftarchive/redis/releases)

---

## Testing the Connection

```python
# test_redis.py
import asyncio
from src.memory.redis_store import RedisMemoryStore, RedisConfig

async def test():
    config = RedisConfig.from_env()
    store = RedisMemoryStore(config)
    
    if await store.connect():
        health = await store.health_check()
        print(f"✅ Connected! Latency: {health['latency_ms']}ms")
        
        # Test user creation
        user_id = await store.get_or_create_user_id(phone_number="+1234567890")
        print(f"User ID: {user_id}")
    else:
        print("❌ Connection failed")
    
    await store.disconnect()

asyncio.run(test())
```

Run with:
```bash
python test_redis.py
```

---

## Integration with Sarah Agent

The session manager integrates automatically. Here's how it works:

```python
# In telephony.py (already integrated)
from src.memory.session_manager import init_session_manager, get_session_manager

# At startup
session_manager = await init_session_manager(llm=agent.llm)

# When call comes in (in handle_media_stream)
context = await session_manager.start_session(
    session_id=stream_sid,
    phone_number=caller_phone  # From Twilio
)

if context.is_returning:
    # User has called before!
    greeting = f"Welcome back! {context.greeting_context}"
    
# During conversation
session_manager.add_turn(stream_sid, "user", user_text)
session_manager.add_turn(stream_sid, "assistant", ai_response)

# When call ends
await session_manager.end_session(stream_sid)
```

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `REDIS_URL` | Full connection URL (overrides below) | - |
| `REDIS_HOST` | Redis server hostname | localhost |
| `REDIS_PORT` | Redis server port | 6379 |
| `REDIS_PASSWORD` | Authentication password | - |
| `REDIS_DB` | Database number | 0 |
| `REDIS_SSL` | Use TLS connection | false |
| `MEMORY_PERSISTENCE` | Enable Redis storage | true |
| `MEMORY_SESSION_TTL_HOURS` | Session expiry time | 24 |
| `MEMORY_PROFILE_TTL_DAYS` | User profile expiry | 90 |

---

## Troubleshooting

### Connection Refused
```
❌ Redis connection failed: Connection refused
```
- Check Redis is running: `redis-cli ping`
- Verify host/port in .env

### Authentication Failed
```
❌ Redis connection failed: NOAUTH Authentication required
```
- Set `REDIS_PASSWORD` in .env
- For cloud: Copy password from dashboard

### SSL/TLS Errors
```
❌ Redis connection failed: SSL handshake failed
```
- Set `REDIS_SSL=true` for cloud services
- For local without SSL: `REDIS_SSL=false`

### Memory Issues
```
OOM command not allowed
```
- Upgrade Redis tier
- Reduce session TTL
- Clear old data: `redis-cli FLUSHDB`
