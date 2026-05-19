# Gemma 4 Customer Support Analysis API

A production-ready FastAPI inference server for Google Gemma 4 E2B-it with three core capabilities:
- **Analyze** — classifies customer support messages into 6 structured outputs
- **Translate** — translates customer messages from any language into English
- **Generate** — free-form text generation

Runs fully containerised on GPU (NVIDIA T4), with Prometheus + Grafana + Loki monitoring and auto-restart on reboot via systemd.

---

## How It Works

### /analyze flow
```
Customer message
      ↓
Preprocessing  (mojibake fix → HTML strip → unicode normalise → signature strip)
      ↓
Brute force check  (10 wrong keys / 60s → IP blocked for 5 min)
      ↓
Concurrency queue  (max 10 active, up to 50 waiting, 429 if full)
      ↓
Gemma 4 E2B-it  (greedy decoding — deterministic JSON output)
      ↓
JSON parser + Pydantic validation
      ↓
{"sentiment": "...", "tone": "...", "urgency": "...",
 "sentiment_score": 0–100, "query_type": "...", "churn_risk": 0–100}
      ↓
Appended to /data/analyze_results.csv
```

### /translate flow
```
Customer message (any language)
      ↓
Preprocessing + auth + concurrency queue  (same as above)
      ↓
Gemma 4 E2B-it  (greedy decoding — low-temperature translation)
      ↓
Plain English text
      ↓
Persistent call count saved to /data/translate_count.txt
```

---

## Project Structure

```
Gemma_Backend_vLLM/
├── app/
│   ├── main.py               # FastAPI app, all endpoints, concurrency queue, CSV logging
│   ├── inference.py          # TransformersBackend / VLLMBackend
│   ├── preprocessing.py      # Text cleaning pipeline
│   ├── auth.py               # API key auth + brute force protection
│   ├── config.py             # Settings (reads from .env)
│   ├── schemas.py            # Pydantic request/response models
│   └── metrics.py            # Prometheus gauges, counters, histograms
├── monitoring/
│   ├── prometheus.yml        # Scrape config (gemma_api, node, gpu)
│   └── promtail.yml          # Log shipper config for Loki
├── tests/
│   ├── test_api.py
│   └── test_auth.py
├── Dockerfile                # Container image for the API
├── docker-compose.yml        # All services in one file (API + monitoring)
├── docker-compose.monitoring.yml  # Monitoring stack only
├── startup.sh                # Boot script (mount disk, start containers) — run by systemd
├── gemma-startup.service     # systemd unit that runs startup.sh on every reboot
├── deploy.sh                 # One-shot fresh-machine deployment script
├── .env.example              # Template — copy to .env and fill in values
└── requirements.txt
```

---

## API Endpoints

### `GET /health`
No auth required.

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
Classifies a customer support message into structured fields.

**Headers:**
```
Content-Type: application/json
X-API-Key: your-api-key
```

**Body:**

| Field | Required | Type | Limit | Description |
|---|---|---|---|---|
| `message` | ✅ Yes | string | 1–5000 chars | Raw customer message |
| `max_new_tokens` | ❌ No | integer | 1–512, default 256 | Leave at default |

**Example:**
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
    "churn_risk": 55,
    "model_warnings": []
  },
  "prompt_tokens": 618,
  "completion_tokens": 49,
  "preprocessed_message": "I have been waiting 3 weeks..."
}
```

> `model_warnings` is an empty list in normal operation. If the model returns an invalid value for a field, it is substituted with a safe default and described in `model_warnings`.

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

### `POST /translate`
Translates a customer message from any language into English.

**Headers:**
```
Content-Type: application/json
X-API-Key: your-api-key
```

**Body:**

| Field | Required | Type | Limit | Description |
|---|---|---|---|---|
| `message` | ✅ Yes | string | 1–5000 chars | Customer message in any language |

**Example:**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/translate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "لم يصل طلبي منذ أسبوعين. أريد استرداد أموالي فوراً"}'
```

**Response:**
```json
{
  "translated_text": "My order has not arrived for two weeks. I want a refund immediately.",
  "prompt_tokens": 75,
  "completion_tokens": 16,
  "preprocessed_message": "لم يصل طلبي منذ أسبوعين. أريد استرداد أموالي فوراً"
}
```

> If the message is already in English it is returned unchanged. Supports all major languages including Arabic, Spanish, French, Urdu, Chinese, German, and more.

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

**Example:**
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
| `503` | Model still loading — wait and retry |
| `500` | Inference error — check logs |

---

## Security

- **API docs disabled** — `/docs`, `/redoc`, `/openapi.json` return 404
- **Brute force protection** — 10 failed auth attempts per 60s blocks the IP for 5 minutes
- **Message size limits** — `/analyze` and `/translate` max 5000 chars, `/generate` max 10,000 chars
- **Metrics auth** — `/metrics` requires the same API key (constant-time comparison)
- **Sanitised errors** — internal exceptions never leak stack traces or model output to callers
- **Server header hidden** — responds as `server: api`
- **CORS** — controlled via `ALLOWED_ORIGINS` env var (set to your domain in production)

---

## Persistent Data

### Analyze CSV
Every successful `/analyze` call is appended to `/data/analyze_results.csv`.

Columns: `timestamp`, `preprocessed_message`, `sentiment`, `tone`, `urgency`, `sentiment_score`, `query_type`, `churn_risk`, `prompt_tokens`, `completion_tokens`, `inference_duration_seconds`

### Translate counter
Every successful `/translate` call increments `/data/translate_count.txt` (a plain integer file).

Both files live on the mounted NVMe drive and are read on container startup to seed Prometheus gauges — so all-time counts survive restarts and never reset to zero.

---

## Cloud Deployment (AWS EC2)

### Recommended instance

| | |
|---|---|
| Instance | `g4dn.xlarge` |
| GPU | NVIDIA T4 (16 GB VRAM) |
| OS | Ubuntu 22.04 LTS |
| Storage | 100 GB NVMe attached at `/data` |

### Scripts — what each one does

| Script | When to use | What it does |
|---|---|---|
| `deploy.sh` | Once, on a fresh machine | Installs Docker, builds the image, writes `.env`, installs the systemd service, starts everything |
| `startup.sh` | Every reboot (run by systemd automatically) | Mounts `/data`, loads the NVIDIA kernel module, starts the monitoring stack and the API container |
| `gemma-startup.service` | Installed once by `deploy.sh` | systemd unit file that calls `startup.sh` on boot — never run directly |

### One-command deploy (fresh machine)

```bash
# 1. Edit the top of deploy.sh — set REPO_URL, HF_TOKEN, API_KEY, GRAFANA_PASSWORD
nano deploy.sh

# 2. Run it
bash deploy.sh
```

`deploy.sh` does everything end-to-end:
- Installs Docker and NVIDIA Container Toolkit
- Clones the repo into `/data/Gemma_Backend_vLLM`
- Writes a `.env` file with your credentials
- Builds the `gemma-api:latest` Docker image
- Installs `gemma-startup.service` into systemd and enables it
- Calls `startup.sh` to start all services immediately

After this, **every reboot is fully automatic**.

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
GRAFANA_PASSWORD=your-grafana-password
```

**3. Build the Docker image**
```bash
docker build -t gemma-api:latest .
```

**4. Install the systemd service**
```bash
sudo cp gemma-startup.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gemma-startup
```

**5. Start everything**
```bash
bash startup.sh
```

### Boot sequence (automatic after setup)

```
Server reboots
      ↓
/data auto-mounts (fstab UUID entry)
      ↓
gemma-startup.service runs startup.sh
→ checks /data mount
→ loads NVIDIA kernel module
→ starts monitoring stack (Prometheus, Grafana, Loki, Node Exporter, DCGM)
→ starts gemma-api container
      ↓
All services live in ~40 seconds (model already cached)
```

---

## Running Services

| Service | URL | Description |
|---|---|---|
| Gemma API | `http://YOUR_IP:8000` | Inference API |
| Grafana | `http://YOUR_IP:3000` | Dashboards |
| Prometheus | `http://YOUR_IP:9090` | Metrics store |
| Loki | `http://YOUR_IP:3100` | Log store |

### Start / stop commands

```bash
# Check what's running
docker ps

# View API logs live
docker logs -f gemma-api

# Restart API only
docker restart gemma-api

# Start monitoring stack
cd /data/Gemma_Backend_vLLM && docker compose -f docker-compose.monitoring.yml up -d

# Stop monitoring stack
docker compose -f docker-compose.monitoring.yml down

# Start all services manually (same as boot)
bash /data/Gemma_Backend_vLLM/startup.sh
```

---

## Monitoring (Grafana)

Login: `admin` / `<GRAFANA_PASSWORD from .env>`

Data sources: `http://prometheus:9090` and `http://loki:3100`

### Prometheus panel queries

| Panel | Visualisation | Query |
|---|---|---|
| API up/down | Stat | `up{job="gemma_api"} * gemma_model_loaded` |
| Total analyze calls (all-time) | Stat | `gemma_analyze_success_total` |
| Total translate calls (all-time) | Stat | `gemma_translate_success_total` |
| Request rate (all endpoints) | Time series | `rate(gemma_requests_total{status="success"}[1m]) * 60` |
| Analyze request rate | Time series | `rate(gemma_requests_total{endpoint="analyze",status="success"}[1m]) * 60` |
| Translate request rate | Time series | `rate(gemma_requests_total{endpoint="translate",status="success"}[1m]) * 60` |
| Avg inference time (all) | Time series | `rate(gemma_inference_duration_seconds_sum[1m]) / rate(gemma_inference_duration_seconds_count[1m])` |
| Translate p50 latency | Time series | `histogram_quantile(0.50, rate(gemma_translate_duration_seconds_bucket[5m]))` |
| Translate p95 latency | Time series | `histogram_quantile(0.95, rate(gemma_translate_duration_seconds_bucket[5m]))` |
| Translate p99 latency | Time series | `histogram_quantile(0.99, rate(gemma_translate_duration_seconds_bucket[5m]))` |
| Active requests | Stat | `gemma_active_requests` |
| Queue depth | Stat | `gemma_queued_requests` |
| Error rate | Time series | `rate(gemma_requests_total{status="error"}[1m]) * 60` |
| GPU utilization | Time series | `DCGM_FI_DEV_GPU_UTIL` |
| GPU memory used | Time series | `DCGM_FI_DEV_FB_USED` |
| CPU usage % | Time series | `100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)` |
| RAM usage % | Time series | `(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100` |

> For the translate p50/p95/p99 panel, add all three queries in a single Time series panel and label them `p50`, `p95`, `p99` in the Legend field.

### Loki query (API logs panel)

```
{job="gemma_api"} |~ "endpoint=/(analyze|translate)"
```

This shows a log line per inference call for both `/analyze` and `/translate`, including duration and token counts.

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
| `API_KEY` | _(required)_ | Auth key — never leave empty in production |
| `ALLOW_NO_AUTH` | `false` | Set `true` only for local dev with no API key |
| `MAX_CONCURRENT` | `10` | Max simultaneous requests |
| `MAX_QUEUE_DEPTH` | `50` | Max requests allowed to queue before 429 |
| `ALLOWED_ORIGINS` | `*` | CORS origins — set to your domain in production |
| `GRAFANA_PASSWORD` | `changeme` | Grafana admin password |
| `TRUST_PROXY` | `false` | Set `true` if behind a reverse proxy (enables X-Forwarded-For) |

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
API_KEY=
ALLOW_NO_AUTH=true
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

> **Never commit your `.env` file.** It is in `.gitignore`. Rotate your `HF_TOKEN` immediately if it has ever been pushed to a public repository.
