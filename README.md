<p align="center">
  <img src="docs/landing/assets/engram-logo.png" alt="engram" width="300">
</p>

<p align="center">
  <strong>Persistent memory for AI agents</strong>
</p>

[![PyPI](https://img.shields.io/pypi/v/engram-mem)](https://pypi.org/project/engram-mem/) ![Tests](https://img.shields.io/badge/tests-996%2B-brightgreen) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)

Dual-memory AI system combining **episodic (vector)** + **semantic (graph)** memory with LLM reasoning. Entity-gated ingestion ensures only meaningful data is stored. Enterprise-ready with multi-tenancy, auth, caching, observability, and Docker deployment.

Works with **any AI agent or IDE** — Claude Code, OpenClaw, Cursor, and any MCP-compatible client. Federates with external knowledge systems (mem0, LightRAG, Graphiti) via auto-discovery. Exposes **CLI**, **MCP** (stdio), **HTTP API** (`/api/v1/`), and **WebSocket** (`/ws`) interfaces.

```bash
pip install engram-mem
```

---

## Features

### Core Memory

- **Episodic Memory** — Qdrant vector store (embedded or server), semantic similarity search, Ebbinghaus decay, activation-based scoring, topic-key upsert
- **Semantic Graph** — NetworkX MultiDiGraph, typed entities and relationships, SQLite (default) or PostgreSQL backend, weighted edges
- **Reasoning Engine** — LLM synthesis (Gemini via litellm), dual-memory context fusion, constitution-guarded prompts
- **Recall Pipeline** — Query decision, temporal+pronoun entity resolution, parallel multi-source search, dedup, composite scoring
- **Entity-Gated Ingestion** — Only stores messages with extracted entities; skips noise (system prompts, trivial messages)
- **Auto Memory** — Detect and persist save-worthy messages automatically, poisoning guard for injection prevention
- **Meeting Ledger** — Structured meeting records with decisions, action items, attendees, topics
- **Feedback Loop** — Confidence scoring (+0.15/-0.2), importance adjustment, auto-delete on 3x negative feedback
- **Graph Visualization** — Interactive entity relationship explorer with dark theme, search, click-to-inspect (vis-network)

### Intelligence Layer

- **Temporal Resolution** — 28 Vietnamese+English date patterns resolve "hom nay/yesterday" to ISO dates before storing
- **Pronoun Resolution** — "anh ay/he/she" to named entity from graph context, LLM-based fallback
- **Fusion Formatter** — Group recall results by type `[preference]`/`[fact]`/`[lesson]` for structured LLM context
- **Memory Consolidation** — Jaccard clustering + LLM summarization reduces redundancy

### Multi-Agent & Federated Knowledge

- **Agent Support** — Claude Code, OpenClaw, Cursor, any MCP-compatible agent or IDE
- **Session Capture** — Real-time JSONL session watchers for OpenClaw + Claude Code (inotify/watchdog)
- **Federated Search** — Query mem0, LightRAG, Graphiti, custom REST/File/Postgres/MCP providers in parallel
- **Auto-Discovery** — Scans local ports, file paths, and MCP configs (`~/.claude/`, `~/.cursor/`) to find providers
- **Provider Adapters** — REST (with JWT auto-login), File (glob patterns), PostgreSQL (custom SQL), MCP (stdio)

### Enterprise

- **Multi-Surface** — CLI (Typer), MCP Server (stdio), HTTP API (FastAPI), WebSocket, Web UI
- **Authentication** — JWT + API keys with RBAC (ADMIN, AGENT, READER), optional, disabled by default
- **Multi-Tenancy** — Isolated per-tenant stores, contextvar propagation, row-level PostgreSQL isolation
- **Caching** — Redis-backed result caching with per-endpoint TTLs
- **Rate Limiting** — Sliding-window per-tenant limits, `fail_open` option
- **Audit Trail** — Structured before/after JSONL log for every episodic mutation
- **Resource Tiers** — 4-tier LLM degradation (FULL > STANDARD > BASIC > READONLY), 60s auto-recovery
- **Data Constitution** — 3-law LLM governance (namespace isolation, no fabrication, audit rights), SHA-256 tamper detection
- **Consolidation Scheduler** — Asyncio background tasks (cleanup daily, consolidate 6h, decay daily), tier-aware
- **Key Rotation** — Failover/round-robin for embedding API keys (GEMINI_API_KEY + GEMINI_API_KEY_FALLBACK)
- **Observability** — OpenTelemetry + JSONL audit logging (optional)
- **Deployment** — Docker Compose, Kubernetes-ready, health checks
- **Backup/Restore** — Memory snapshots, point-in-time recovery
- **Benchmark Suite** — p50/p95/p99 latency measurements for all endpoints

---

## Architecture

```mermaid
flowchart TD
    subgraph Agents["Agents & IDEs"]
        CC["Claude Code"]
        OC["OpenClaw"]
        CU["Cursor"]
        ANY["Any MCP Client"]
    end

    subgraph Interfaces
        CLI["CLI (Typer)"]
        MCP["MCP (stdio)"]
        HTTP["HTTP API /api/v1/"]
        WS["WebSocket /ws"]
    end

    CC & OC & CU & ANY --> MCP
    CLI & MCP & HTTP & WS --> Auth["Auth Middleware\n(JWT + RBAC, optional)"]
    Auth --> Tenant["TenantContext (ContextVar)"]
    Tenant --> Recall["Recall Pipeline\n(decision > resolve > search > feedback)"]
    Recall --> Episodic["EpisodicStore\n(Qdrant)"]
    Recall --> Semantic["SemanticGraph\n(NetworkX + SQLite/PG)"]
    Recall --> Fed["Federated Providers"]
    Episodic & Semantic --> Reasoning["Reasoning Engine\n(Gemini via litellm)"]
    Episodic --> Cache["Redis Cache (optional)"]
    WS --> EventBus["Event Bus\n(push events)"]

    subgraph Fed["Federated Knowledge"]
        M0["mem0"]
        LR["LightRAG"]
        GR["Graphiti"]
        REST["REST / File / PG / MCP"]
    end
```

---

## Quick Start

```bash
# Install from PyPI
pip install engram-mem

# Or from source
git clone https://github.com/docaohieu2808/Engram-Mem.git
cd engram && pip install -e .

# Initialize config
engram init

# Set API key
export GEMINI_API_KEY="your-key"

# Start daemon (background HTTP server + watcher)
engram start

# Store a memory
engram remember "Deployed v2.1 to production at 14:00 - caused 503 spike"

# Search memories
engram recall "production incidents"

# Browse all data (episodic + semantic)
engram dump

# Reason across all memory
engram think "What deployment issues have we had?"
```

**Requirements:** Python 3.11+, `GEMINI_API_KEY` for LLM reasoning and embeddings. Basic storage works without it.

---

## Integrations

### Claude Code (MCP)

Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram-mcp",
      "env": { "GEMINI_API_KEY": "your-key" }
    }
  }
}
```

### Cursor (MCP)

Add to Cursor's MCP settings — engram auto-discovers Cursor's config at `~/.cursor/settings.json`:

```json
{
  "mcpServers": {
    "engram": {
      "command": "engram-mcp",
      "env": { "GEMINI_API_KEY": "your-key" }
    }
  }
}
```

### OpenClaw

Install the engram skill, then enable session watcher in `~/.engram/config.yaml`:

```yaml
capture:
  openclaw:
    enabled: true
    sessions_dir: ~/.openclaw/workspace/sessions
```

### Federated Knowledge Providers

Engram auto-discovers and federates with external memory systems. Supported providers:

| Provider | Type | Auto-Discovery |
|----------|------|----------------|
| **mem0** | REST | Port 8080, `/v1/memories` |
| **LightRAG** | REST | Port 9520, `/query` |
| **Graphiti** | REST | Port 8000, `/search` |
| **OpenClaw** | File | `~/.openclaw/workspace/memory/*.md` |
| **Custom REST** | REST | Manual config |
| **PostgreSQL** | SQL | Manual config |
| **MCP servers** | MCP | Scans `~/.claude/settings.json`, `~/.cursor/settings.json` |

```yaml
# Auto-discovery (enabled by default)
discovery:
  local: true
  hosts: ["10.10.0.2"]  # additional hosts to scan

# Or manual provider config
providers:
  - name: my-mem0
    type: rest
    url: http://localhost:8080
    search_endpoint: /v1/memories/search
    search_method: POST
    search_body: '{"query": "{query}", "limit": {limit}}'
    result_path: "results[].memory"
```

### HTTP API

```bash
# Start server
engram serve --port 8765

# Store memory
curl -X POST http://localhost:8765/api/v1/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "Deployed v1.0", "memory_type": "fact", "priority": 8}'

# Search
curl "http://localhost:8765/api/v1/recall?query=deployment&limit=5"

# Reason
curl -X POST http://localhost:8765/api/v1/think \
  -H "Content-Type: application/json" \
  -d '{"question": "What deployment issues have we had?"}'

# Meeting ledger
curl -X POST http://localhost:8765/api/v1/meeting-ledger \
  -H "Content-Type: application/json" \
  -d '{"title": "Sprint Review", "decisions": ["Ship v2"], "action_items": ["Update docs"]}'
```

---

## CLI Reference (61 Commands)

### Memory Operations
```bash
engram remember <content> [--type fact|decision|...] [--priority 1-10]
                          [--tags tag1,tag2] [--expires 7d] [--topic-key key]
engram recall <query> [--limit 5] [--type <type>] [--tags tag1,tag2]
engram ask <question>               # Smart query (auto-routes)
engram think <question>             # LLM reasoning
engram summarize [--count 20] [--save]
engram decay [--limit 20]           # Ebbinghaus retention curve
```

### Semantic Graph
```bash
engram add node <name> --type <type>
engram add edge <from> <to> --relation <relation>
engram remove node <key>
engram remove edge <key>
engram query [keyword] [--type X] [--related-to Y] [--format table|json]
engram autolink-orphans [--apply] [--min-co-mentions 3]
```

### Browse & Export
```bash
engram status                       # Memory counts
engram dump [--format table|json]   # All memories + graph
engram health                       # Full system health check
engram tui                          # Terminal UI (interactive browser)
engram graph [--port 8100]          # Open visualization browser
```

### Data Management
```bash
engram cleanup                      # Delete expired memories
engram consolidate [--limit 50]     # LLM clustering + summarization
engram ingest <file.json> [--dry-run]  # Extract entities + remember
engram backup                       # Export snapshot
engram restore <file>               # Import snapshot
engram migrate <file>               # Import legacy JSON
```

### Session & Feedback
```bash
engram session-start
engram session-end
engram feedback <id> --positive|--negative
engram resolve <query>              # Pronoun + temporal resolution
engram audit [--limit 50]           # Retrieval audit log
```

### Server & Capture
```bash
engram init                         # Zero-config setup
engram start                        # Start daemon (HTTP server + watcher)
engram stop                         # Stop daemon
engram logs [--tail 50]             # Show logs
engram serve [--host 0.0.0.0] [--port 8765]  # Foreground HTTP server
engram watch [--daemon]             # Watch inbox + OpenClaw/Claude Code sessions
```

### Configuration & Setup
```bash
engram setup                        # Interactive IDE connector wizard
engram config show|get <key>|set <key> <value>
engram auth                         # API key management
engram providers discover           # Auto-discover external providers
engram providers list|add|remove    # Manage providers
engram schema                       # Manage semantic schemas
```

### Monitoring & Status
```bash
engram queue-status                 # Embedding queue health
engram resource-status              # LLM tier (FULL/STANDARD/BASIC/READONLY)
engram constitution-status          # 3-law governance + SHA-256
engram scheduler-status             # Background task schedule
engram benchmark [--quick]          # Run recall accuracy benchmark
```

### Daemon & Advanced
```bash
engram autostart                    # Install systemd user services
engram sync [--direction]           # Git-friendly memory sharing
```

---

## MCP Tools (21 Total)

| Tool | Description |
|------|-------------|
| `engram_remember` | Store episodic memory with type, priority, tags, expires, topic-key |
| `engram_recall` | Search episodic memories (compact or full) with filtering |
| `engram_get_memory` | Retrieve full memory content by ID or 8-char prefix |
| `engram_timeline` | Get chronological context around a memory (±window minutes) |
| `engram_cleanup` | Delete all expired memories |
| `engram_cleanup_dedup` | Deduplicate similar memories by cosine similarity threshold |
| `engram_ingest` | Dual ingest: extract entities + store memories from chat |
| `engram_feedback` | Record positive/negative feedback (adjusts confidence) |
| `engram_auto_feedback` | Auto-detect feedback sentiment from text |
| `engram_think` | Reason across episodic + semantic memory via LLM |
| `engram_ask` | Smart query — auto-routes to recall or think based on intent |
| `engram_summarize` | Summarize recent N memories into insights via LLM |
| `engram_add_entity` | Add/update entity node to knowledge graph |
| `engram_add_relation` | Add/update relationship edge between entities |
| `engram_query_graph` | Query knowledge graph (keyword, type, related-to) |
| `engram_meeting_ledger` | Record structured meeting (decisions, action items, attendees) |
| `engram_status` | Show memory statistics (episodic count, semantic nodes/edges) |
| `engram_session_start` | Begin new conversation session |
| `engram_session_end` | End active session |
| `engram_session_summary` | Get summary of completed session |
| `engram_session_context` | Retrieve memories from active session |

---

## Configuration

**Config file:** `~/.engram/config.yaml` — Priority: CLI flags > env vars > YAML > defaults

```yaml
episodic:
  mode: embedded              # embedded (Qdrant in-process) or server
  path: ~/.engram/qdrant
  namespace: default

embedding:
  provider: gemini
  model: gemini-embedding-001
  key_strategy: failover      # failover or round-robin

semantic:
  provider: sqlite            # or postgresql
  path: ~/.engram/semantic.db

llm:
  provider: gemini
  model: gemini/gemini-2.0-flash
  api_key: ${GEMINI_API_KEY}

serve:
  host: 127.0.0.1
  port: 8765

capture:
  openclaw:
    enabled: false
    sessions_dir: ~/.openclaw/workspace/sessions
  claude_code:
    enabled: false
    sessions_dir: ~/.claude/projects

auth:
  enabled: false
cache:
  enabled: false
  redis_url: redis://localhost:6379/0
rate_limit:
  enabled: false
audit:
  enabled: false
  path: ~/.engram/audit.jsonl
```

---

## API Reference

Start server: `engram serve [--host 0.0.0.0] [--port 8765]`

**Health & Info:**

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Liveness check |
| GET | `/health/ready` | Readiness probe |
| GET | `/graph` | Interactive graph UI |

**Core Operations** (`/api/v1/`):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/remember` | Store episodic memory |
| GET | `/recall` | Search memories (`?query=X&limit=5`) |
| POST | `/think` | LLM reasoning across episodic + semantic |
| GET | `/query` | Graph search (`?keyword=X&node_type=Y&related_to=Z`) |
| POST | `/ingest` | Extract entities + store memories |
| POST | `/meeting-ledger` | Record structured meeting |
| POST | `/feedback` | Record memory feedback |

**Memory Management** (`/api/v1/`):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/memories` | List/filter with pagination |
| GET | `/memories/{id}` | Get single memory |
| PUT | `/memories/{id}` | Update memory |
| DELETE | `/memories/{id}` | Delete memory |
| GET | `/memories/export` | Export all as JSON |
| POST | `/memories/bulk-delete` | Batch delete |

**Semantic Graph** (`/api/v1/`):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/graph/data` | Graph data (nodes + edges) for vis.js |
| POST | `/graph/nodes` | Add/update node |
| PUT | `/graph/nodes/{key}` | Update node |
| DELETE | `/graph/nodes/{key}` | Delete node |
| POST | `/graph/edges` | Add/update edge |
| DELETE | `/graph/edges` | Delete edge |
| GET | `/feedback/history` | Feedback history |

**Admin** (`/api/v1/`):

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/cleanup` | Delete expired memories |
| POST | `/cleanup/dedup` | Deduplicate memories |
| POST | `/auth/token` | Get JWT token |
| GET | `/providers` | List active providers |
| GET | `/audit/log` | Retrieval audit log |
| GET | `/scheduler/tasks` | Scheduler status |
| POST | `/scheduler/tasks/{name}/run` | Run task now |
| POST | `/benchmark/run` | Run benchmark |
| GET | `/config` | Get config |
| PUT | `/config` | Update config |
| GET | `/status` | Memory statistics |

---

## WebSocket API

Connect via `ws://host:8765/ws?token=JWT` (token optional when auth disabled).

**Commands:**

| Command | Payload |
|---------|---------|
| `remember` | `{"content": "...", "priority": 7}` |
| `recall` | `{"query": "...", "limit": 5}` |
| `think` | `{"question": "..."}` |
| `feedback` | `{"memory_id": "abc123", "feedback": "positive"}` |
| `query` | `{"keyword": "PostgreSQL"}` |
| `ingest` | `{"messages": [...]}` |
| `status` | `{}` |

**Push Events:** `memory_created`, `memory_updated`, `memory_deleted`, `feedback_recorded`

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | LLM + embeddings (primary key) |
| `GEMINI_API_KEY_FALLBACK` | Secondary key for key rotation |
| `ENGRAM_NAMESPACE` | Memory namespace isolation |
| `ENGRAM_AUTH_ENABLED` | Enable JWT auth |
| `ENGRAM_SEMANTIC_PROVIDER` | `sqlite` or `postgresql` |
| `ENGRAM_CACHE_ENABLED` | Enable Redis caching |
| `ENGRAM_AUDIT_ENABLED` | Enable audit logs |
| `ENGRAM_TELEMETRY_ENABLED` | Enable OpenTelemetry |

---

## Docker

```bash
# Quick start
docker build -t engram:latest .
docker run -e GEMINI_API_KEY="your-key" -p 8765:8765 engram:latest

# Production with PostgreSQL + Redis
ENGRAM_AUTH_ENABLED=true \
ENGRAM_SEMANTIC_PROVIDER=postgresql \
ENGRAM_SEMANTIC_DSN=postgresql://user:pass@postgres:5432/engram \
ENGRAM_CACHE_ENABLED=true \
ENGRAM_CACHE_REDIS_URL=redis://redis:6379/0 \
docker compose up
```

---

## Testing

```bash
pytest tests/ -v                      # All tests
pytest tests/ --cov=src/engram        # With coverage
pytest tests/ -k "recall or feedback" # Specific suites
```

894+ tests, 61%+ code coverage, CI/CD via GitHub Actions.

---

## Documentation

- [Project Overview & PDR](docs/project-overview-pdr.md)
- [System Architecture](docs/system-architecture.md)
- [Code Standards](docs/code-standards.md)
- [Deployment Guide](docs/deployment-guide.md)
- [Codebase Summary](docs/codebase-summary.md)
- [Project Roadmap](docs/project-roadmap.md)
- [Changelog](docs/project-changelog.md)

---

## License

MIT — Copyright (c) Do Cao Hieu
