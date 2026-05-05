#!/bin/bash
# =============================================================================
# Gemma 4 API — Full Cloud Deployment Script
# Tested on: Ubuntu 22.04 / AWS EC2 (g4dn.xlarge recommended)
# Run as: bash deploy.sh
# =============================================================================

set -e

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
section() { echo -e "\n${BOLD}━━━  $*  ━━━${RESET}"; }

# =============================================================================
# CONFIG — edit these before running
# =============================================================================
DATA_MOUNT="/data"                         # mounted volume path
APP_DIR="$DATA_MOUNT/Gemma_Backend_vLLM"  # where the code lives
VENV_DIR="$DATA_MOUNT/gemma-env"          # Python venv on the data drive
HF_CACHE="$DATA_MOUNT/hf_cache"           # model cache on the data drive
LOG_FILE="$DATA_MOUNT/api.log"
DOCKER_DATA="$DATA_MOUNT/docker"
PYTHON_BIN="python3.11"                   # must be ≥3.11
API_PORT=8000
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

REPO_URL=""          # set to your git repo URL, or leave empty to skip git clone
HF_TOKEN=""          # set your HuggingFace token here, or export HF_TOKEN before running
API_KEY=""           # leave empty to disable auth, or set a strong secret

# =============================================================================
# STEP 0 — Pre-flight checks
# =============================================================================
section "Pre-flight checks"

[[ "$EUID" -eq 0 ]] && error "Do not run as root. Run as ubuntu (sudo access is fine)."

# Resolve HF_TOKEN — env var takes precedence over inline
[[ -z "$HF_TOKEN" && -n "${HF_TOKEN:-}" ]] && HF_TOKEN="${HF_TOKEN}"
[[ -z "$HF_TOKEN" ]] && warn "HF_TOKEN is not set — model download will fail on gated models."

# Check data mount
if ! mountpoint -q "$DATA_MOUNT" 2>/dev/null; then
    warn "$DATA_MOUNT is not a separate mount point — using root disk."
fi
info "Data drive: $(df -h $DATA_MOUNT | awk 'NR==2{print $2" total, "$4" free"}')"

# =============================================================================
# STEP 1 — System packages
# =============================================================================
section "System packages"

sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3.11 python3.11-venv python3-pip \
    git curl wget unzip \
    nginx \
    build-essential
success "System packages installed"

# =============================================================================
# STEP 2 — Data disk layout
# =============================================================================
section "Data disk layout"

sudo mkdir -p "$DATA_MOUNT"
mkdir -p "$HF_CACHE" "$DOCKER_DATA"
success "Directories ready on $DATA_MOUNT"

# =============================================================================
# STEP 3 — Clone or update the repo
# =============================================================================
section "Code deployment"

if [[ -n "$REPO_URL" ]]; then
    if [[ -d "$APP_DIR/.git" ]]; then
        info "Repo exists — pulling latest changes"
        git -C "$APP_DIR" pull
    else
        info "Cloning repo into $APP_DIR"
        git clone "$REPO_URL" "$APP_DIR"
    fi
    success "Code at $APP_DIR"
elif [[ -d "$APP_DIR" ]]; then
    info "REPO_URL not set — using existing code at $APP_DIR"
else
    error "No repo URL set and $APP_DIR does not exist. Set REPO_URL at the top of this script."
fi

# =============================================================================
# STEP 4 — Python virtual environment
# =============================================================================
section "Python virtual environment"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    info "Creating venv at $VENV_DIR"
    $PYTHON_BIN -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip -q
info "Installing Python dependencies"
pip install -q -r "$APP_DIR/requirements.txt"

# Install vLLM only if CUDA is available
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    info "GPU detected — installing vLLM"
    pip install -q vllm
    INFERENCE_BACKEND="vllm"
    success "vLLM installed — will use GPU backend"
else
    warn "No GPU detected — using Transformers (CPU) backend"
    INFERENCE_BACKEND="transformers"
fi

deactivate
success "Python environment ready"

# =============================================================================
# STEP 5 — .env file
# =============================================================================
section ".env configuration"

ENV_FILE="$APP_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    cp "$APP_DIR/.env.example" "$ENV_FILE"
    info "Created .env from .env.example"
fi

# Generate API key if not provided
if [[ -z "$API_KEY" ]]; then
    API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    warn "No API_KEY set — generated one automatically"
fi

# Write all values
cat > "$ENV_FILE" <<EOF
# Auto-generated by deploy.sh on $(date)
MODEL_ID=google/gemma-4-E2B-it
HF_TOKEN=$HF_TOKEN

INFERENCE_BACKEND=$INFERENCE_BACKEND

MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
LOAD_IN_4BIT=true

HOST=0.0.0.0
PORT=$API_PORT

API_KEY=$API_KEY

MAX_CONCURRENT=10
MAX_QUEUE_DEPTH=50
EOF

chmod 600 "$ENV_FILE"
success ".env written to $ENV_FILE"
info "API_KEY: $API_KEY"

# =============================================================================
# STEP 6 — Pre-download the model
# =============================================================================
section "Model download"

if [[ -z "$HF_TOKEN" ]]; then
    warn "Skipping model pre-download — HF_TOKEN not set"
else
    info "Pre-downloading model to $HF_CACHE (this may take a while on first run)"
    HF_HOME="$HF_CACHE" "$VENV_DIR/bin/python" -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    'google/gemma-4-E2B-it',
    token=os.environ.get('HF_TOKEN', ''),
    cache_dir='$HF_CACHE',
    ignore_patterns=['*.msgpack','*.h5','flax_model*','tf_model*'],
)
print('Model download complete')
" && success "Model cached at $HF_CACHE" || warn "Model download failed — will retry on first API start"
fi

# =============================================================================
# STEP 7 — systemd service
# =============================================================================
section "systemd service"

sudo tee /etc/systemd/system/gemma-api.service > /dev/null <<EOF
[Unit]
Description=Gemma 4 Customer Support API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
Environment=HF_HOME=$HF_CACHE
ExecStart=$VENV_DIR/bin/uvicorn app.main:app --host 0.0.0.0 --port $API_PORT
Restart=on-failure
RestartSec=10
StandardOutput=append:$LOG_FILE
StandardError=append:$LOG_FILE

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable gemma-api
sudo systemctl restart gemma-api
success "gemma-api service started"

# =============================================================================
# STEP 8 — Docker + Prometheus + Grafana
# =============================================================================
section "Docker + Prometheus + Grafana"

if ! command -v docker &>/dev/null; then
    info "Installing Docker"
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
        sudo tee /etc/apt/keyrings/docker.asc > /dev/null
    sudo chmod a+r /etc/apt/keyrings/docker.asc
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
        sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
    sudo usermod -aG docker ubuntu
    success "Docker installed"
else
    info "Docker already installed: $(docker --version)"
fi

# Point Docker data-root to the data drive
if ! grep -q '"data-root"' /etc/docker/daemon.json 2>/dev/null; then
    echo "{\"data-root\": \"$DOCKER_DATA\"}" | sudo tee /etc/docker/daemon.json > /dev/null
    sudo systemctl restart docker
    info "Docker data-root set to $DOCKER_DATA"
fi

info "Starting Prometheus + Grafana"
cd "$APP_DIR"
sudo docker compose -f docker-compose.monitoring.yml up -d
success "Prometheus and Grafana started"

# =============================================================================
# STEP 9 — Wait for API to come up and verify
# =============================================================================
section "Health check"

info "Waiting for API to be ready (up to 60s)..."
for i in $(seq 1 12); do
    sleep 5
    HEALTH=$(curl -s http://localhost:$API_PORT/health 2>/dev/null)
    if echo "$HEALTH" | grep -q '"status":"ok"'; then
        success "API is up"
        echo "$HEALTH"
        break
    fi
    info "Attempt $i/12 — waiting..."
done

if ! echo "$HEALTH" | grep -q '"status":"ok"'; then
    warn "API did not respond within 60s — check logs: sudo journalctl -u gemma-api -n 50"
fi

# =============================================================================
# STEP 10 — Summary
# =============================================================================
section "Deployment complete"

PUBLIC_IP=$(curl -sf http://checkip.amazonaws.com || curl -sf http://ifconfig.me || echo "<your-server-ip>")

echo -e "
${BOLD}┌─────────────────────────────────────────────────────┐
│              GEMMA 4 API — LIVE                     │
├─────────────────────────────────────────────────────┤${RESET}
  API Health    : http://$PUBLIC_IP:$API_PORT/health
  API Docs      : http://$PUBLIC_IP:$API_PORT/docs
  Analyze       : POST http://$PUBLIC_IP:$API_PORT/analyze
  Prometheus    : http://$PUBLIC_IP:$PROMETHEUS_PORT
  Grafana       : http://$PUBLIC_IP:$GRAFANA_PORT  (admin/admin)
  Logs          : $LOG_FILE
  Live logs     : sudo journalctl -u gemma-api -f
${BOLD}├─────────────────────────────────────────────────────┤${RESET}
  API_KEY       : $API_KEY
  Backend       : $INFERENCE_BACKEND
  HF Cache      : $HF_CACHE
${BOLD}└─────────────────────────────────────────────────────┘${RESET}

${YELLOW}Make sure ports $API_PORT, $PROMETHEUS_PORT, $GRAFANA_PORT are open in your EC2 Security Group.${RESET}
"
