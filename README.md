# Gemma 4 Customer Support Analysis API

A FastAPI inference server for Google Gemma 4 E2B-it that classifies customer support messages into 6 structured outputs: sentiment, tone, urgency, sentiment score, query type, and churn risk.

---

## How It Works

```
Customer message
      ↓
Preprocessing  (mojibake fix → HTML strip → unicode normalise → signature strip)
      ↓
Concurrency queue  (max 10 active, up to 50 waiting, 429 if full)
      ↓
Gemma 4 E2B-it  (Transformers backend locally, vLLM on cloud GPU)
      ↓
JSON parser + Pydantic validation
      ↓
{"sentiment": "...", "tone": "...", "urgency": "...",
 "sentiment_score": 0-100, "query_type": "...", "churn_risk": 0-100}
```

---

## Project Structure

```
Gemma_Backend_vLLM/
├── app/
│   ├── main.py          # FastAPI app, endpoints, concurrency queue
│   ├── inference.py     # TransformersBackend / VLLMBackend
│   ├── preprocessing.py # Text cleaning pipeline
│   ├── auth.py          # API key authentication
│   ├── config.py        # Settings (reads from .env)
│   ├── schemas.py       # Pydantic request/response models
│   └── metrics.py       # Prometheus gauges and counters
├── tests/
│   ├── test_analyze.py
│   └── test_auth.py
├── monitoring/          # Grafana/Prometheus docker-compose configs
├── .env.example         # Template — copy to .env and fill in values
├── requirements.txt
└── start.sh
```

---

## Local Setup (Windows / No GPU)

### 1. Clone and install

```bash
git clone <repo-url>
cd Gemma_Backend_vLLM
py -3.13 -m pip install -r requirements.txt
```

### 2. Create your `.env`

```bash
copy .env.example .env
```

Open `.env` and set at minimum:

```
HF_TOKEN=hf_your_token_here
INFERENCE_BACKEND=transformers
API_KEY=                        # leave empty for local dev
```

### 3. Run

```bash
py -3.13 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Or use the start script:

```bash
bash start.sh
```

---

## API Key Authentication

### How it works

- If `API_KEY` is empty in `.env` → **auth is disabled** (safe for local dev)
- If `API_KEY` is set → every request to `/generate` and `/analyze` **must** include the header:

```
X-API-Key: your-secret-key
```

Requests without it, or with the wrong key, receive `401 Unauthorized`.

The `/health` and `/metrics` endpoints are always public.

### Setting up on a new server

**Step 1 — generate a strong key**

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
# example output: a3f8c2d1e9b047f6a2c5d8e3f1b4a7c2d5e8f1b4a7c2d5e8f1b4a7c2d5e8f1b4
```

**Step 2 — add it to `.env` on the server**

```bash
nano .env
# set: API_KEY=a3f8c2d1e9b047f6a2c5d8e3f1b4a7c2...
```

**Step 3 — restart the server**

```bash
sudo systemctl restart gemma-api
```

**Step 4 — test it**

```bash
# Should return 401
curl http://your-server/analyze -X POST -H "Content-Type: application/json" \
  -d '{"message": "test"}'

# Should return 200
curl http://your-server/analyze -X POST \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key-here" \
  -d '{"message": "My order is late!"}'
```

> **Never commit your `.env` file.** It is listed in `.gitignore`. Rotate your `HF_TOKEN` if it has ever been pushed to a repository.

---

## Concurrency & Queue

The server processes at most **10 requests simultaneously**. Additional requests are queued (up to 50 waiting). If the queue is also full, the server returns `429 Too Many Requests`.

```
Incoming request
    │
    ├─ Queue full (>50 waiting)? → 429
    │
    └─ Add to queue
           │
           └─ Wait for a slot (max 10 active)
                  │
                  └─ Process → return response
```

You can tune these values in `.env`:

```
MAX_CONCURRENT=10    # raise if you have a bigger GPU / more memory
MAX_QUEUE_DEPTH=50   # raise if you expect burst traffic
```

Current queue state is visible at `/health`:

```json
{
  "status": "ok",
  "model_loaded": true,
  "active_requests": 3,
  "queued_requests": 7,
  "max_concurrent": 10,
  "max_queue_depth": 50
}
```

---

## API Reference

### `GET /health`

No auth required. Returns server and queue status.

### `POST /generate`

Requires `X-API-Key` header (if configured).

```json
{
  "prompt": "Tell me about...",
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### `POST /analyze`

Requires `X-API-Key` header (if configured).

```json
{ "message": "My order has not arrived after 3 weeks." }
```

Response:

```json
{
  "result": {
    "sentiment": "negative",
    "tone": "frustrated",
    "urgency": "high",
    "sentiment_score": 28,
    "query_type": "shipping-delay",
    "churn_risk": 65
  },
  "prompt_tokens": 614,
  "completion_tokens": 52,
  "raw_response": "...",
  "preprocessed_message": "My order has not arrived after 3 weeks."
}
```

**Sentiment:** `positive` | `neutral` | `negative` | `threatening`

**Tone:** `calm` | `frustrated` | `angry` | `anxious` | `appreciative` | `demanding` | `sarcastic`

**Urgency:** `low` | `medium` | `high` | `critical`

**Query type:** `order-status` | `shipping-delay` | `address-change` | `order-modification` | `order-hold` | `refund` | `billing` | `product-inquiry` | `restock-inquiry` | `damaged-item` | `warranty` | `technical-issue` | `general-inquiry`

### `GET /metrics`

Prometheus metrics endpoint. No auth required.

---

## Cloud Deployment (AWS EC2)

### Recommended instance

| | |
|---|---|
| Instance | `g4dn.xlarge` |
| GPU | NVIDIA T4 (16 GB VRAM) |
| OS | Ubuntu 22.04 LTS |
| Cost | ~$0.53/hr on-demand |

### Steps

**1. Connect to your instance**

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@YOUR_EC2_IP
```

**2. Install dependencies**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git nginx
```

**3. Upload your code**

From your local machine:

```bash
scp -i your-key.pem -r C:\Users\Nysonian\Documents\Gemma_Backend_vLLM ubuntu@YOUR_EC2_IP:~/
```

**4. Set up Python environment**

```bash
cd ~/Gemma_Backend_vLLM
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Uncomment vllm in requirements.txt first if using vLLM backend
```

**5. Configure environment**

```bash
cp .env.example .env
nano .env
```

Set:
```
HF_TOKEN=hf_your_token
INFERENCE_BACKEND=vllm         # use vllm on Linux+GPU
API_KEY=your-generated-secret
```

**6. Create systemd service**

```bash
sudo nano /etc/systemd/system/gemma-api.service
```

```ini
[Unit]
Description=Gemma 4 Support API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/Gemma_Backend_vLLM
EnvironmentFile=/home/ubuntu/Gemma_Backend_vLLM/.env
ExecStart=/home/ubuntu/Gemma_Backend_vLLM/.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable gemma-api
sudo systemctl start gemma-api
sudo systemctl status gemma-api
```

**7. Nginx reverse proxy**

```bash
sudo nano /etc/nginx/sites-available/gemma-api
```

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_read_timeout 300s;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/gemma-api /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

**8. TLS (HTTPS)**

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Useful commands on the server

```bash
# View live logs
sudo journalctl -u gemma-api -f

# Restart after code changes
sudo systemctl restart gemma-api

# Check queue state
curl http://localhost:8000/health
```

---

## Monitoring

Prometheus + Grafana are included via Docker Compose:

```bash
docker compose -f docker-compose.monitoring.yml up -d
```

Key metrics:

| Metric | What it tells you |
|---|---|
| `gemma_active_requests` | Requests being processed right now |
| `gemma_queued_requests` | Requests waiting for a slot |
| `gemma_inference_duration_seconds` | How long each request takes |
| `gemma_tokens_per_second` | Throughput |
| `gemma_requests_total` | Success / error counts |

---

## Running Tests

```bash
py -3.13 -m pytest tests/ -v
```

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `google/gemma-4-E2B-it` | HuggingFace model ID |
| `HF_TOKEN` | _(required)_ | HuggingFace access token |
| `INFERENCE_BACKEND` | `transformers` | `transformers` (local) or `vllm` (cloud) |
| `MAX_NEW_TOKENS` | `512` | Max tokens to generate |
| `TEMPERATURE` | `0.7` | Sampling temperature |
| `TOP_P` | `0.9` | Nucleus sampling threshold |
| `LOAD_IN_4BIT` | `true` | 4-bit quantization (requires CUDA + bitsandbytes) |
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `API_KEY` | _(empty)_ | Auth key — empty disables auth |
| `MAX_CONCURRENT` | `10` | Max simultaneous requests |
| `MAX_QUEUE_DEPTH` | `50` | Max requests allowed to queue |
