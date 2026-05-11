# Gemma 4 Customer Support Analysis API — Full Technical Documentation

---

## 1. Overview

### What This Service Is

This is a production-ready AI inference API built to automatically classify customer support messages. When a customer sends a message to a support team, this API reads that message and returns a structured analysis in under 6 seconds.

The analysis tells you:
- How the customer is feeling (sentiment)
- What tone they are using (tone)
- How urgently they need a response (urgency)
- A numeric score of their emotional state (sentiment_score)
- What kind of issue they have (query_type)
- How likely they are to leave as a customer (churn_risk)

This removes the need for a human agent to manually triage and prioritize every incoming message.

### What It Is Built With

| Component | Technology |
|---|---|
| API framework | FastAPI (Python) |
| AI model | Google Gemma 4 E2B-it (5.1 billion parameters) |
| Inference engine | HuggingFace Transformers |
| GPU | NVIDIA Tesla T4 (16 GB VRAM) |
| Containerization | Docker |
| Metrics | Prometheus + Grafana |
| Logs | Loki + Promtail |
| Host OS | Ubuntu 22.04 on AWS EC2 (g4dn.xlarge) |
| Auto-start | systemd |

### Live Endpoints (Current Deployment)

| Service | URL |
|---|---|
| Gemma API | http://13.127.136.155:8000 |
| Grafana Dashboard | http://13.127.136.155:3000 |
| Prometheus | http://13.127.136.155:9090 |
| Loki (logs) | http://13.127.136.155:3100 |

---

## 2. System Architecture

### High-Level Flow

```
User / Application
        |
        | HTTP POST with JSON + API Key
        v
+------------------+
|   FastAPI App    |  Port 8000
|                  |
|  1. Auth check   |  -- wrong key 10x in 60s → IP blocked 5 min
|  2. Queue check  |  -- >50 waiting → 429 Too Many Requests
|  3. Preprocess   |  -- clean text, strip HTML, fix encoding
|  4. Prompt build |  -- insert message into analysis prompt template
|  5. Inference    |  -- send to Gemma 4 model on GPU
|  6. Parse JSON   |  -- extract and validate model output
|  7. CSV log      |  -- append row to /data/analyze_results.csv
|  8. Return       |  -- send structured JSON response
+------------------+
        |
        | Metrics scraped every 15s
        v
+------------------+      +------------------+
|   Prometheus     |----->|    Grafana       |
+------------------+      +------------------+
        |
        v
+------------------+
|   Loki + Promtail|  (log storage and search)
+------------------+
```

### Concurrency Model

The API handles up to 10 requests simultaneously on the GPU. Additional requests are held in a queue.

```
Incoming request
    |
    +-- Is queue > 50?  → Return 429 immediately
    |
    +-- Add to queue, wait for a GPU slot
           |
           +-- GPU slot free → process request (~5 seconds)
           |
           +-- Return JSON response
```

This prevents the GPU from being overwhelmed and ensures every request that enters the queue gets a response.

### Boot Sequence (After Server Reboot)

```
EC2 instance powers on
        |
        v
/data NVMe disk auto-mounts  (via /etc/fstab UUID entry)
        |
        v
containerd starts  (systemd)
        |
        v
Docker daemon starts  (systemd)
        |
        v
Docker sees 7 containers with "unless-stopped" policy
→ Restarts all 7 automatically
        |
        v
gemma-startup.service fires  (systemd, runs startup.sh)
→ Loads NVIDIA kernel module
→ Starts monitoring stack via docker compose
→ Starts gemma-api container
→ Health check passes
        |
        v
All services live in approximately 40 seconds
No manual action required
```

---

## 3. File Structure — What Each File Does

```
Gemma_Backend_vLLM/
├── app/
│   ├── main.py
│   ├── inference.py
│   ├── preprocessing.py
│   ├── auth.py
│   ├── config.py
│   ├── schemas.py
│   └── metrics.py
├── monitoring/
│   ├── prometheus.yml
│   └── promtail.yml
├── tests/
│   ├── test_analyze.py
│   └── test_auth.py
├── Dockerfile
├── docker-compose.yml
├── docker-compose.monitoring.yml
├── startup.sh
├── gemma-startup.service
├── deploy.sh
├── .env
├── .env.example
└── requirements.txt
```

### app/main.py
The core of the application. Contains:
- FastAPI app initialization with docs disabled (security)
- Two API endpoints: /analyze and /generate
- Concurrency queue logic using asyncio Semaphore
- Middleware to strip the server identity header
- Protected /metrics endpoint (requires API key)
- CSV logging — after every successful /analyze call, appends a row to /data/analyze_results.csv
- The analysis prompt template that instructs the model

### app/inference.py
Handles loading the AI model and running inference. Contains two backend classes:
- **TransformersBackend** — uses HuggingFace Transformers library, works on both CPU and GPU, used in production on the T4 GPU
- **VLLMBackend** — uses vLLM for faster batched inference (not currently compatible with Gemma 4 on T4 due to a shared memory limitation in the Triton attention kernel)

The correct backend is selected automatically based on the INFERENCE_BACKEND setting in .env.

### app/preprocessing.py
Cleans raw customer messages before they are sent to the model. The pipeline does:
1. **Mojibake fix** — corrects garbled text caused by wrong character encoding (e.g. â€™ becomes ')
2. **HTML entity decode** — converts &amp; to &, &lt; to <, etc.
3. **HTML tag strip** — removes any HTML tags
4. **Unicode normalize** — converts lookalike characters to standard form (e.g. ﬁ → fi)
5. **Whitespace normalize** — collapses tabs, non-breaking spaces, excessive blank lines
6. **Reply thread strip** — removes "On [date] wrote:" email thread history
7. **Signature strip** — removes email signatures, phone numbers, job titles at the end of messages

### app/auth.py
Handles API key authentication and brute force protection.
- Reads the correct API key from .env
- Checks the X-API-Key header on every protected request
- Tracks failed attempts per IP address in memory
- After 10 failed attempts within 60 seconds, blocks that IP for 5 minutes
- Successful authentication clears the failed attempt history for that IP

### app/config.py
Reads all settings from the .env file using Pydantic Settings. Any environment variable defined in .env is automatically loaded here and available throughout the app as settings.variable_name.

### app/schemas.py
Defines the shape of all request and response data using Pydantic models. Enforces:
- Message length limits (1–5000 characters for /analyze, 1–10,000 for /generate)
- Valid values for sentiment, tone, urgency, query_type
- Numeric ranges for sentiment_score and churn_risk (0–100)
- If the model returns an invalid value, Pydantic silently corrects it to a safe default

### app/metrics.py
Defines all Prometheus metrics that the API tracks:
- **gemma_inference_duration_seconds** — histogram of how long each request takes
- **gemma_prompt_tokens_total** — total tokens sent to the model
- **gemma_completion_tokens_total** — total tokens generated by the model
- **gemma_tokens_per_second** — throughput histogram
- **gemma_active_requests** — how many requests are on the GPU right now
- **gemma_queued_requests** — how many requests are waiting
- **gemma_requests_total** — total count by endpoint (analyze/generate) and status (success/error)
- **gemma_model_loaded** — 1 when model is ready, 0 when not

### monitoring/prometheus.yml
Configuration for Prometheus. Tells it where to scrape metrics from every 15 seconds:
- **gemma_api** — scrapes http://host.docker.internal:8000/metrics/ with API key auth
- **node** — scrapes node-exporter for CPU, RAM, disk metrics
- **gpu** — scrapes dcgm-exporter for GPU utilization and memory

### monitoring/promtail.yml
Configuration for Promtail (log shipper). Tails /data/api.log and ships each line to Loki. Also extracts the endpoint, status, and inference_duration fields from log lines as structured labels for querying in Grafana.

### Dockerfile
Instructions to build the API into a Docker image. Uses NVIDIA CUDA 12.6 as the base image (required for GPU access). Installs Python 3.11, all Python dependencies from requirements.txt, and copies the app code. The image is approximately 8–10 GB due to PyTorch.

### docker-compose.yml
Defines all 7 services in one file so they can be started together:
- gemma-api (the inference API)
- prometheus
- grafana
- loki
- promtail
- node-exporter
- dcgm-exporter (GPU metrics)

### docker-compose.monitoring.yml
The monitoring stack only (without gemma-api). Used by startup.sh.

### startup.sh
Shell script that runs on every server boot. Steps:
1. Checks if /data is mounted, mounts it if not
2. Restores the /var/lib/containerd symlink (points to /data/containerd to save root disk space)
3. Waits for Docker to be ready
4. Loads the NVIDIA kernel module
5. Starts the monitoring stack
6. Removes any stale gemma-api container
7. Starts the gemma-api Docker container with GPU access
8. Runs a health check

### gemma-startup.service
A systemd unit file. Tells the Linux boot system to run startup.sh automatically after the network and Docker are ready on every reboot. This is what makes the service fully hands-off after initial setup.

To install: copy to /etc/systemd/system/ and run:
```bash
sudo systemctl daemon-reload
sudo systemctl enable gemma-startup
```

### deploy.sh
A one-shot deployment script for fresh EC2 instances. Automates the entire setup from a clean Ubuntu 22.04 machine: installs packages, formats and mounts the NVMe disk, clones the repo, creates the Python environment, downloads the model, sets up systemd, and starts all services.

### .env
The main configuration file. Contains secrets and runtime settings. This file is never committed to git. Must be created manually on each deployment from .env.example.

### .env.example
A template of the .env file with all variables listed but no real values. Commit this to git so other developers know what variables are needed.

---

## 4. API Reference

### Authentication

All endpoints except /health require an API key passed as a header:

```
X-API-Key: your-api-key-here
```

Requests with a missing or wrong key return HTTP 401.
After 10 failed attempts from the same IP within 60 seconds, that IP is blocked for 5 minutes and receives HTTP 429.

---

### GET /health

No authentication required. Returns the current status of the server.

**Request:**
```bash
curl http://YOUR_SERVER_IP:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "active_requests": 2,
  "queued_requests": 0
}
```

| Field | Description |
|---|---|
| status | Always "ok" if the server is running |
| model_loaded | true when the AI model is loaded and ready |
| active_requests | Number of requests currently being processed |
| queued_requests | Number of requests waiting for a GPU slot |

---

### POST /analyze

Analyzes a customer support message and returns structured classification data.

**Headers:**
```
Content-Type: application/json
X-API-Key: your-api-key
```

**Request body:**

| Field | Required | Type | Constraints | Description |
|---|---|---|---|---|
| message | Yes | string | 1–5000 characters | The raw customer message to analyze |
| max_new_tokens | No | integer | 1–512, default 256 | Maximum length of model response — leave empty |

**Example:**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"message": "I have been waiting 3 weeks for my order and nobody is responding. This is completely unacceptable!"}'
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
  "raw_response": "{\"sentiment\": \"negative\", ...}",
  "preprocessed_message": "I have been waiting 3 weeks for my order..."
}
```

**Result field values:**

| Field | Possible Values | Description |
|---|---|---|
| sentiment | positive, neutral, negative, threatening | Overall emotional direction. threatening overrides all others when the customer mentions legal action, chargebacks, or public exposure |
| tone | calm, frustrated, angry, anxious, appreciative, demanding, sarcastic | Delivery style of the message |
| urgency | low, medium, high, critical | How quickly the message needs a response. critical means deadline within 48 hours or threatening sentiment |
| sentiment_score | 0–100 | Numeric score. 0–20 threatening, 21–45 negative, 46–54 neutral, 55–100 positive |
| query_type | order-status, shipping-delay, address-change, order-modification, order-hold, refund, billing, product-inquiry, restock-inquiry, damaged-item, warranty, technical-issue, general-inquiry | The category of the customer's issue. If multiple issues, picks the most urgent or irreversible |
| churn_risk | 0–100 | Likelihood the customer will leave. Baseline 50. Increases for threatening language, refund requests, repeated issues. Decreases for positive sentiment |

---

### POST /generate

Free-form text generation. Send any prompt and receive a generated response.

**Request body:**

| Field | Required | Type | Constraints | Description |
|---|---|---|---|---|
| prompt | Yes | string | 1–10,000 characters | Any instruction or text for the model |
| max_new_tokens | No | integer | 1–4096, default 512 | Maximum length of the response |
| temperature | No | float | 0.0–2.0, default 0.7 | Creativity. Lower = more predictable. Higher = more creative |
| top_p | No | float | 0.0–1.0, default 0.9 | Controls vocabulary variety. Leave at default |

**Example:**
```bash
curl -X POST http://YOUR_SERVER_IP:8000/generate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"prompt": "Write a polite reply to a customer whose order is 3 weeks late."}'
```

**Response:**
```json
{
  "response": "Dear valued customer, we sincerely apologize for the delay with your order...",
  "prompt_tokens": 45,
  "completion_tokens": 134
}
```

---

### GET /metrics

Prometheus metrics endpoint. Returns all internal performance counters. Requires API key.

```bash
curl http://YOUR_SERVER_IP:8000/metrics/ \
  -H "X-API-Key: your-api-key"
```

---

### Error Codes

| HTTP Code | Meaning | When It Happens |
|---|---|---|
| 200 | Success | Request processed correctly |
| 401 | Unauthorized | Missing or wrong API key |
| 422 | Unprocessable Entity | Message too long, empty, or wrong data type |
| 429 | Too Many Requests | Queue is full, or IP blocked after brute force attempts |
| 503 | Service Unavailable | Model is still loading — retry in a few seconds |

---

## 5. Security

The following security measures are implemented:

### API Key Authentication
Every request to /analyze, /generate, and /metrics must include the X-API-Key header. Requests without it receive HTTP 401.

### Brute Force Protection
The auth layer tracks failed key attempts per IP address. After 10 failures within a 60-second window, the IP is blocked for 5 minutes and receives HTTP 429 with a Retry-After header. A successful login clears the failure history for that IP.

### Input Size Limits
- /analyze accepts a maximum of 5,000 characters
- /generate accepts a maximum of 10,000 characters
Requests exceeding these limits are rejected immediately with HTTP 422 before reaching the model. This prevents a single oversized request from occupying a GPU worker slot.

### API Docs Disabled
FastAPI automatically generates interactive documentation at /docs and /redoc. These are disabled in production. They return HTTP 404. This prevents attackers from getting a complete map of the API.

### Server Header Removed
By default, the server reveals it is running on uvicorn (server: uvicorn). A middleware replaces this with server: api, hiding the technology stack from potential attackers.

### CORS Restriction
Cross-Origin Resource Sharing is controlled by the ALLOWED_ORIGINS environment variable. In production this should be set to your specific frontend domain (e.g. https://yourdomain.com). The current setting of * means any website can call the API from a browser.

### Metrics Protected
The /metrics endpoint previously had no authentication and exposed internal performance data. It now requires the same API key as all other endpoints. Prometheus scrapes it using a Bearer token in the Authorization header.

### Known Limitation — Prompt Injection
Because the API uses a language model, a sufficiently crafted customer message can manipulate the model's output. This is a known limitation of all LLM-based systems. The risk is mitigated by strict Pydantic output validation (invalid field values are silently corrected to safe defaults) and a low inference temperature (0.1) which reduces model creativity.

---

## 6. Data Collection

Every successful /analyze call appends one row to /data/analyze_results.csv on the mounted NVMe drive.

**File location:** /data/analyze_results.csv

**Columns:**

| Column | Description |
|---|---|
| timestamp | UTC datetime in ISO 8601 format |
| preprocessed_message | The customer message after cleaning |
| sentiment | Classification result |
| tone | Classification result |
| urgency | Classification result |
| sentiment_score | 0–100 numeric score |
| query_type | Classification result |
| churn_risk | 0–100 numeric score |
| prompt_tokens | Tokens sent to the model |
| completion_tokens | Tokens generated by the model |
| inference_duration_seconds | How long the request took |

This file is never deleted automatically and grows over time. It can be downloaded and analyzed in Excel or Python.

---

## 7. Monitoring

### Services

| Service | URL | Purpose |
|---|---|---|
| Grafana | http://YOUR_IP:3000 | Visual dashboards — login: admin / admin |
| Prometheus | http://YOUR_IP:9090 | Metrics database |
| Loki | http://YOUR_IP:3100 | Log database |
| Node Exporter | http://YOUR_IP:9100 | Host CPU/RAM/disk metrics |
| DCGM Exporter | http://YOUR_IP:9400 | GPU metrics |

### Setting Up Grafana

1. Open http://YOUR_IP:3000 and log in (admin / admin)
2. Go to Connections → Data sources → Add new data source
3. Add Prometheus with URL: http://prometheus:9090
4. Add Loki with URL: http://loki:3100
5. Go to Dashboards → New → New dashboard → Add visualization

### Key Grafana Queries

**API health (shows 1=online, 0=offline):**
```
up{job="gemma_api"} * gemma_model_loaded
```

**Requests per minute:**
```
rate(gemma_requests_total{status="success"}[1m]) * 60
```

**Analyze requests per minute only:**
```
rate(gemma_requests_total{endpoint="analyze",status="success"}[1m]) * 60
```

**Average inference time in seconds:**
```
rate(gemma_inference_duration_seconds_sum[1m]) / rate(gemma_inference_duration_seconds_count[1m])
```

**Active requests on GPU:**
```
gemma_active_requests
```

**Requests waiting in queue:**
```
gemma_queued_requests
```

**Error rate per minute:**
```
rate(gemma_requests_total{status="error"}[1m]) * 60
```

**GPU utilization percentage:**
```
DCGM_FI_DEV_GPU_UTIL
```

**GPU memory used (MiB):**
```
DCGM_FI_DEV_FB_USED
```

**CPU usage percentage:**
```
100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)
```

**RAM usage percentage:**
```
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100
```

**Log panel (Loki data source):**
```
{job="gemma_api"} |= "inference_duration"
```

---

## 8. Running on Any Machine

### Option A — Docker (Recommended for Any Linux Server with GPU)

**Requirements:**
- Docker installed
- NVIDIA Container Toolkit installed
- NVIDIA GPU with CUDA support

**Steps:**

```bash
# 1. Clone the repository
git clone <repo-url>
cd Gemma_Backend_vLLM

# 2. Create the environment file
cp .env.example .env
nano .env
# Fill in HF_TOKEN and API_KEY at minimum

# 3. Build the Docker image
docker build -t gemma-api:latest .

# 4. Run the API
docker run -d \
  --name gemma-api \
  --gpus all \
  -p 8000:8000 \
  --env-file .env \
  -v /data/hf_cache:/data/hf_cache \
  -v /data:/data \
  --restart unless-stopped \
  gemma-api:latest \
  sh -c "uvicorn app.main:app --host 0.0.0.0 --port 8000 2>&1 | tee -a /data/api.log"

# 5. Start monitoring
docker compose -f docker-compose.monitoring.yml up -d

# 6. Check it is running
curl http://localhost:8000/health
```

**All services at once (including API):**
```bash
docker compose up -d
```

---

### Option B — Local Development (Windows or Mac, No GPU Required)

**Requirements:**
- Python 3.11
- 16+ GB RAM (model runs on CPU, will be slow — ~90 seconds per request)

**Steps:**

```bash
# 1. Clone the repository
git clone <repo-url>
cd Gemma_Backend_vLLM

# 2. Create virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create environment file
# Windows:
copy .env.example .env
# Mac/Linux:
cp .env.example .env

# Open .env and set:
# HF_TOKEN=hf_your_token_here
# INFERENCE_BACKEND=transformers
# API_KEY=   (leave empty for local dev — auth disabled)

# 5. Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Test it
curl http://localhost:8000/health
```

---

### Option C — AWS EC2 from Scratch (Full Production Setup)

**Recommended instance:** g4dn.xlarge (1x T4 GPU, 4 vCPU, 16 GB RAM, ~$0.53/hr)

```bash
# 1. Connect to your instance
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# 2. Clone and run the deploy script
git clone <repo-url>
cd Gemma_Backend_vLLM
bash deploy.sh
```

The deploy script handles everything automatically. After it completes, all services start on every reboot with no manual action needed.

**Manual AWS setup if not using deploy.sh:**

```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install -y nvidia-driver-535

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker ubuntu

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Mount the NVMe data disk
sudo mkfs.ext4 /dev/nvme1n1
sudo mkdir /data
sudo mount /dev/nvme1n1 /data
echo "UUID=$(sudo blkid -s UUID -o value /dev/nvme1n1)  /data  ext4  defaults,nofail  0  2" | sudo tee -a /etc/fstab

# Clone repo to /data
git clone <repo-url> /data/Gemma_Backend_vLLM
cd /data/Gemma_Backend_vLLM
cp .env.example .env
nano .env  # fill in values

# Build and start
docker build -t gemma-api:latest .
docker compose up -d

# Register systemd auto-start
sudo cp gemma-startup.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable gemma-startup
```

---

## 9. Environment Variables Reference

The .env file controls all runtime behavior. Copy .env.example to .env and fill in values before starting.

| Variable | Default | Required | Description |
|---|---|---|---|
| MODEL_ID | google/gemma-4-E2B-it | No | HuggingFace model identifier |
| HF_TOKEN | (empty) | Yes | HuggingFace access token for downloading the model |
| INFERENCE_BACKEND | transformers | No | Use "transformers" always (vllm not compatible with Gemma 4 on T4) |
| MAX_NEW_TOKENS | 512 | No | Default maximum tokens the model generates per request |
| TEMPERATURE | 0.7 | No | Response creativity. 0.1 used for /analyze (consistent), 0.7 for /generate |
| TOP_P | 0.9 | No | Nucleus sampling. Leave at default |
| LOAD_IN_4BIT | false | No | 4-bit quantization. Keep false — bitsandbytes CUDA build not available |
| HOST | 0.0.0.0 | No | Network interface to bind to |
| PORT | 8000 | No | Port number |
| API_KEY | (empty) | Yes (production) | Secret key for API authentication. Leave empty for local dev only |
| MAX_CONCURRENT | 10 | No | How many requests the GPU processes at the same time |
| MAX_QUEUE_DEPTH | 50 | No | How many requests can wait before returning 429 |
| ALLOWED_ORIGINS | * | No | CORS allowed origins. Set to your frontend domain in production |

---

## 10. Useful Commands

### Check all services are running
```bash
docker ps
curl http://localhost:8000/health
```

### View live API logs
```bash
docker logs -f gemma-api
# or
tail -f /data/api.log
```

### Restart the API only
```bash
docker restart gemma-api
```

### Stop everything
```bash
cd /data/Gemma_Backend_vLLM
docker compose down
sudo lsof -ti:8000 | xargs sudo kill -9
```

### Start everything
```bash
cd /data/Gemma_Backend_vLLM
docker compose up -d
```

### View the CSV results
```bash
cat /data/analyze_results.csv
```

### Check GPU status
```bash
nvidia-smi
```

### Check disk space on /data
```bash
df -h /data
```

### Run tests
```bash
cd /data/Gemma_Backend_vLLM
source /data/gemma-env/bin/activate
pytest tests/ -v
```

---

## 11. Known Limitations

**vLLM not compatible with Gemma 4 on T4**
The vLLM inference engine (which offers faster concurrent batching) requires 96KB of shared memory per streaming multiprocessor for Gemma 4's heterogeneous attention heads. The T4 only provides 64KB. This causes a crash at startup. The Transformers backend is used instead and provides ~5 second inference time, which is acceptable.

**Prompt injection possible**
A crafted customer message can sometimes influence the model's classification output. The Pydantic validation layer catches invalid field values, but numeric fields (sentiment_score, churn_risk) can still be manipulated within valid ranges. This is an inherent limitation of instruction-following language models.

**CORS still wide open**
The ALLOWED_ORIGINS variable is currently set to * in .env. This should be updated to the specific frontend domain before sharing the API with external users.

**In-memory brute force tracking**
The failed authentication attempt counter is stored in application memory. It resets when the container restarts. A production system would use Redis for persistent tracking across restarts.

**Model runs in float16 without quantization**
The model uses approximately 10 GB of GPU VRAM, leaving ~5 GB free. 4-bit quantization (which would reduce this to ~3 GB) is not available because the bitsandbytes library requires a CUDA build that is not installed in the current environment.

---

*Document generated from live production deployment on AWS EC2 g4dn.xlarge — May 2026*
