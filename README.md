# Gemma 4 Customer Support Analysis API

A production-ready FastAPI inference server for Google Gemma 4 E2B-it that classifies customer support messages into 6 structured outputs: sentiment, tone, urgency, sentiment score, query type, and churn risk.

Runs fully containerized on GPU (NVIDIA T4), with Prometheus + Grafana + Loki monitoring and auto-restart on reboot via systemd.

---

## How It Works

```
Customer message
      ↓
Preprocessing  (mojibake fix → HTML strip → unicode normalise → signature strip)
      ↓
Brute force check  (10 wrong keys / 60s → IP blocked for 5 min)
      ↓
Concurrency queue  (max 10 active, up to 50 waiting, 429 if full)
      ↓
Gemma 4 E2B-it  (Transformers backend on GPU)
      ↓
JSON parser + Pydantic validation
      ↓
{"sentiment": "...", "tone": "...", "urgency": "...",
 "sentiment_score": 0–100, "query_type": "...", "churn_risk": 0–100}
      ↓
Appended to /data/analyze_results.csv
```

---

## Project Structure

```
Gemma_Backend_vLLM/
├── app/
│   ├── main.py               # FastAPI app, endpoints, concurrency queue, CSV logging
│   ├── inference.py          # TransformersBackend / VLLMBackend
│   ├── preprocessing.py      # Text cleaning pipeline
│   ├── auth.py               # API key auth + brute force protection
│   ├── config.py             # Settings (reads from .env)
│   ├── schemas.py            # Pydantic request/response models
│   └── metrics.py            # Prometheus gauges and counters
├── monitoring/
│   ├── prometheus.yml        # Scrape config (gemma_api, node, gpu)
│   └── promtail.yml          # Log shipper config for Loki
├── tests/
│   ├── test_analyze.py
│   └── test_auth.py
├── Dockerfile                # Container image for the API
├── docker-compose.yml        # All services in one file (API + monitoring)
├── docker-compose.monitoring.yml  # Monitoring stack only
├── startup.sh                # Boot script (mount disk, start containers)
├── gemma-startup.service     # systemd unit — runs startup.sh on every reboot
├── deploy.sh                 # One-shot cloud deployment script
├── .env.example              # Template — copy to .env and fill in values
└── requirements.txt
```

---

## API Endpoints

### `GET /health`
No auth required. Returns server status.

```bash
curl http://YOUR_SERVER_IP:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "active_requests": 0,
  "queued_requests": 0
}
```

---

### `POST /analyze`
Classifies a customer support message.

**Headers:**
```
Content-Type: application/json
X-API-Key: your-api-key
```

**Body:**

| Field | Required | Type | Limit | Description |
|---|---|---|---|---|
| `message` | ✅ Yes | string | 1–5000 chars | Raw customer message |
| `max_new_tokens` | ❌ No | integer | 1–512, default 256 | Max response length — leave it out |

**Example request:**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "I have been waiting 3 weeks for my order and nobody is responding!"}'
```

**Response:**
```json
{
  "result": {
    "sentiment": "negative",
    "tone": "frustrated",
    "urgency": "high",
    "sentiment_score": 36,
    "query_type": "order-status",
    "churn_risk": 55
  },
  "prompt_tokens": 618,
  "completion_tokens": 49,
  "raw_response": "...",
  "preprocessed_message": "I have been waiting 3 weeks..."
}
```

**Field values:**

| Field | Possible values |
|---|---|
| `sentiment` | `positive` `neutral` `negative` `threatening` |
| `tone` | `calm` `frustrated` `angry` `anxious` `appreciative` `demanding` `sarcastic` |
| `urgency` | `low` `medium` `high` `critical` |
| `sentiment_score` | `0–100` (0 = most hostile, 100 = most positive) |
| `query_type` | `order-status` `shipping-delay` `address-change` `order-modification` `order-hold` `refund` `billing` `product-inquiry` `restock-inquiry` `damaged-item` `warranty` `technical-issue` `general-inquiry` |
| `churn_risk` | `0–100` (0 = no risk, 100 = very likely to leave) |

---

### `POST /generate`
Free-form text generation.

**Body:**

| Field | Required | Type | Limit | Description |
|---|---|---|---|---|
| `prompt` | ✅ Yes | string | 1–10,000 chars | Instruction or text for the model |
| `max_new_tokens` | ❌ No | integer | 1–4096, default 512 | Max response length |
| `temperature` | ❌ No | float | 0.0–2.0, default 0.7 | Creativity — lower = more predictable |
| `top_p` | ❌ No | float | 0.0–1.0, default 0.9 | Word variety — leave at default |

**Example request:**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Write a polite reply to a customer who has been waiting 3 weeks for their order."}'
```

**Response:**
```json
{
  "response": "Dear customer, we sincerely apologize for the delay...",
  "prompt_tokens": 45,
  "completion_tokens": 120
}
```

---

### `GET /metrics`
Prometheus metrics. Requires API key.

```bash
curl http://YOUR_SERVER_IP:8000/metrics/ \
  -H "X-API-Key: your-api-key"
```

---

## Error Codes

| Code | Meaning |
|---|---|
| `401` | Missing or wrong API key |
| `422` | Bad request — message too long, empty, or wrong field type |
| `429` | Queue full, or IP blocked after too many wrong key attempts |
| `503` | Model still loading — wait a few seconds and retry |

---

## Security

- **API docs disabled** — `/docs`, `/redoc`, `/openapi.json` return 404 in production
- **Brute force protection** — 10 failed auth attempts per 60s blocks the IP for 5 minutes
- **Message size limit** — `/analyze` max 5000 chars, `/generate` max 10,000 chars
- **Metrics auth** — `/metrics` requires the same API key
- **Server header hidden** — responds as `server: api`, not `server: uvicorn`
- **CORS** — controlled via `ALLOWED_ORIGINS` env var (set to your frontend domain in production)

---

## CSV Output

Every successful `/analyze` call is appended to `/data/analyze_results.csv` on the mounted drive.

Columns: `timestamp`, `preprocessed_message`, `sentiment`, `tone`, `urgency`, `sentiment_score`, `query_type`, `churn_risk`, `prompt_tokens`, `completion_tokens`, `inference_duration_seconds`

---

## Cloud Deployment (AWS EC2)

### Recommended instance

| | |
|---|---|
| Instance | `g4dn.xlarge` |
| GPU | NVIDIA T4 (16 GB VRAM) |
| OS | Ubuntu 22.04 LTS |
| Storage | 100 GB NVMe attached at `/data` |

### One-command deploy

```bash
bash deploy.sh
```

This script handles: system packages, disk mount, Python venv, `.env` setup, model pre-download, Docker + monitoring stack, systemd service, and health check.

### Manual steps

**1. Clone the repo**
```bash
git clone <repo-url> /data/Gemma_Backend_vLLM
cd /data/Gemma_Backend_vLLM
```

**2. Create `.env`**
```bash
cp .env.example .env
nano .env
```

Minimum required:
```
HF_TOKEN=hf_your_token_here
INFERENCE_BACKEND=transformers
API_KEY=your-strong-secret-key
ALLOWED_ORIGINS=https://yourdomain.com
```

**3. Build and run with Docker**
```bash
docker build -t gemma-api:latest .
docker compose up -d
```

**4. Register the startup service**
```bash
sudo cp gemma-startup.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gemma-startup
```

From this point, all services start automatically on every reboot.

### Boot sequence (automatic after setup)

```
Server reboots
      ↓
/data auto-mounts (fstab UUID entry)
      ↓
Docker daemon starts → restarts all containers (unless-stopped policy)
      ↓
gemma-startup.service runs startup.sh
→ loads NVIDIA module
→ starts monitoring stack
→ starts gemma-api container
      ↓
All services live in ~40 seconds
```

---

## Running Services

| Service | URL | Description |
|---|---|---|
| Gemma API | `http://YOUR_IP:8000` | Inference API |
| Grafana | `http://YOUR_IP:3000` | Dashboards (admin / admin) |
| Prometheus | `http://YOUR_IP:9090` | Metrics store |
| Loki | `http://YOUR_IP:3100` | Log store |

### Start / stop commands

```bash
# Start everything
cd /data/Gemma_Backend_vLLM && docker compose up -d

# Stop everything
docker compose down

# Restart API only
docker restart gemma-api

# View API logs live
docker logs -f gemma-api

# Check all container status
docker ps
```

---

## Monitoring (Grafana)

Data sources: `http://prometheus:9090` and `http://loki:3100`

### Key Prometheus queries

| Panel | Query |
|---|---|
| API up/down | `up{job="gemma_api"} * gemma_model_loaded` |
| Request rate | `rate(gemma_requests_total{status="success"}[1m]) * 60` |
| Analyze requests only | `rate(gemma_requests_total{endpoint="analyze",status="success"}[1m]) * 60` |
| Avg inference time | `rate(gemma_inference_duration_seconds_sum[1m]) / rate(gemma_inference_duration_seconds_count[1m])` |
| Active requests | `gemma_active_requests` |
| Queue depth | `gemma_queued_requests` |
| Error rate | `rate(gemma_requests_total{status="error"}[1m]) * 60` |
| GPU utilization | `DCGM_FI_DEV_GPU_UTIL` |
| GPU memory used | `DCGM_FI_DEV_FB_USED` |
| CPU usage % | `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)` |
| RAM usage % | `(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100` |

### Loki query (logs panel)

```
{job="gemma_api"} |= "inference_duration"
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `google/gemma-4-E2B-it` | HuggingFace model ID |
| `HF_TOKEN` | _(required)_ | HuggingFace access token |
| `INFERENCE_BACKEND` | `transformers` | `transformers` or `vllm` |
| `MAX_NEW_TOKENS` | `512` | Default max tokens to generate |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `TOP_P` | `0.9` | Nucleus sampling threshold |
| `LOAD_IN_4BIT` | `false` | 4-bit quantization (requires bitsandbytes CUDA build) |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `API_KEY` | _(empty)_ | Auth key — empty disables auth (local dev only) |
| `MAX_CONCURRENT` | `10` | Max simultaneous requests |
| `MAX_QUEUE_DEPTH` | `50` | Max requests allowed to queue before 429 |
| `ALLOWED_ORIGINS` | `*` | CORS origins — set to your domain in production |

---

## Local Setup (Windows / No GPU)

```bash
git clone <repo-url>
cd Gemma_Backend_vLLM
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then edit .env
py -3.11 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Set in `.env` for local:
```
INFERENCE_BACKEND=transformers
API_KEY=                        # leave empty — auth disabled locally
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

> **Never commit your `.env` file.** It is in `.gitignore`. Rotate your `HF_TOKEN` immediately if it has ever been pushed to a public repository.
