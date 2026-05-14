#!/bin/bash
# =============================================================================
# Gemma 4 API — Full Cloud Deployment Script
# Tested on: Ubuntu 22.04 / AWS EC2 (g4dn.xlarge)
# Run once on a fresh machine:  bash deploy.sh
#
# What this script does:
#   1. Installs system packages (Docker, NVIDIA Container Toolkit)
#   2. Sets up data disk directories
#   3. Clones the repo (or uses existing code)
#   4. Creates .env with your credentials
#   5. Builds the gemma-api Docker image
#   6. Installs gemma-startup.service (systemd unit that boots everything)
#   7. Starts all services via startup.sh
# =============================================================================

set -e

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
DATA_MOUNT="/data"
APP_DIR="$DATA_MOUNT/Gemma_Backend_vLLM"
HF_CACHE="$DATA_MOUNT/hf_cache"
LOG_FILE="$DATA_MOUNT/api.log"
DOCKER_DATA="$DATA_MOUNT/docker"
API_PORT=8000
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

REPO_URL=""      # your git repo URL, e.g. https://github.com/you/Gemma_Backend_vLLM
HF_TOKEN=""      # HuggingFace access token (or export HF_TOKEN before running)
API_KEY=""       # leave empty to auto-generate a secure random key
GRAFANA_PASSWORD="changeme"   # change this before deploying

# =============================================================================
# STEP 0 — Pre-flight
# =============================================================================
section "Pre-flight checks"

[[ "$EUID" -eq 0 ]] && error "Do not run as root. Run as ubuntu (sudo is fine)."

# HF_TOKEN: inline value takes precedence, then env var
[[ -z "$HF_TOKEN" && -n "${HF_TOKEN:-}" ]] && HF_TOKEN="${HF_TOKEN}"
[[ -z "$HF_TOKEN" ]] && warn "HF_TOKEN is not set — model download will fail for gated models."

if ! mountpoint -q "$DATA_MOUNT" 2>/dev/null; then
    warn "$DATA_MOUNT is not a separate mount point — using root disk."
fi
info "Data drive: $(df -h $DATA_MOUNT | awk 'NR==2{print $2" total, "$4" free"}')"

# =============================================================================
# STEP 1 — System packages
# =============================================================================
section "System packages"

sudo apt-get update -qq
sudo apt-get install -y -qq curl git

# Docker
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
    success "Docker installed — you may need to log out/in for group membership"
else
    info "Docker already installed: $(docker --version)"
fi

# NVIDIA Container Toolkit (GPU support for Docker)
if command -v nvidia-smi &>/dev/null && ! dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
    info "Installing NVIDIA Container Toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    success "NVIDIA Container Toolkit installed"
else
    info "NVIDIA Container Toolkit already installed or no GPU found — skipping"
fi

success "System packages ready"

# =============================================================================
# STEP 2 — Data disk layout
# =============================================================================
section "Data disk layout"

mkdir -p "$HF_CACHE" "$DOCKER_DATA"
touch "$LOG_FILE"

# Point Docker data-root to the data drive (keeps Docker images off root disk)
if ! grep -q '"data-root"' /etc/docker/daemon.json 2>/dev/null; then
    echo "{\"data-root\": \"$DOCKER_DATA\"}" | sudo tee /etc/docker/daemon.json > /dev/null
    sudo systemctl restart docker
    info "Docker data-root → $DOCKER_DATA"
fi

success "Directories ready on $DATA_MOUNT"

# =============================================================================
# STEP 3 — Clone or update the repo
# =============================================================================
section "Code"

if [[ -n "$REPO_URL" ]]; then
    if [[ -d "$APP_DIR/.git" ]]; then
        info "Repo exists — pulling latest"
        git -C "$APP_DIR" pull
    else
        info "Cloning repo into $APP_DIR"
        git clone "$REPO_URL" "$APP_DIR"
    fi
    success "Code at $APP_DIR"
elif [[ -d "$APP_DIR" ]]; then
    info "REPO_URL not set — using existing code at $APP_DIR"
else
    error "No REPO_URL set and $APP_DIR does not exist. Set REPO_URL at the top of this script."
fi

# =============================================================================
# STEP 4 — .env file
# =============================================================================
section ".env configuration"

ENV_FILE="$APP_DIR/.env"

# Generate API key if not provided
if [[ -z "$API_KEY" ]]; then
    API_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    warn "No API_KEY provided — generated one automatically. Save it!"
fi

cat > "$ENV_FILE" <<EOF
# Generated by deploy.sh on $(date)
MODEL_ID=google/gemma-4-E2B-it
HF_TOKEN=$HF_TOKEN

INFERENCE_BACKEND=transformers

MAX_NEW_TOKENS=512
TEMPERATURE=0.7
TOP_P=0.9
LOAD_IN_4BIT=false

HOST=0.0.0.0
PORT=$API_PORT

API_KEY=$API_KEY
ALLOW_NO_AUTH=false

MAX_CONCURRENT=10
MAX_QUEUE_DEPTH=50

ALLOWED_ORIGINS=*

GRAFANA_PASSWORD=$GRAFANA_PASSWORD
EOF

chmod 600 "$ENV_FILE"
success ".env written (chmod 600) — API_KEY: $API_KEY"

# =============================================================================
# STEP 5 — Build the Docker image
# =============================================================================
section "Docker image build"

info "Building gemma-api:latest from $APP_DIR/Dockerfile"
cd "$APP_DIR"
sudo docker build -t gemma-api:latest .
success "gemma-api:latest built"

# =============================================================================
# STEP 6 — Install gemma-startup.service
# =============================================================================
section "systemd service"

# gemma-startup.service runs startup.sh on every reboot.
# startup.sh handles: /data mount, NVIDIA module, monitoring stack, gemma-api container.
sudo cp "$APP_DIR/gemma-startup.service" /etc/systemd/system/gemma-startup.service
sudo systemctl daemon-reload
sudo systemctl enable gemma-startup
success "gemma-startup.service installed and enabled"

# =============================================================================
# STEP 7 — First boot: run startup.sh now
# =============================================================================
section "Starting all services"

info "Running startup.sh to bring everything up for the first time"
bash "$APP_DIR/startup.sh"

# =============================================================================
# STEP 8 — Health check
# =============================================================================
section "Health check"

info "Waiting for API (up to 3 min — model needs to load)..."
for i in $(seq 1 18); do
    sleep 10
    HEALTH=$(curl -s http://localhost:$API_PORT/health 2>/dev/null)
    if echo "$HEALTH" | grep -q '"model_loaded":true'; then
        success "API is up and model is loaded"
        echo "$HEALTH"
        break
    fi
    info "Attempt $i/18 — $(echo "$HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); print('model_loaded='+str(d.get('model_loaded',False)))" 2>/dev/null || echo 'waiting...')"
done

if ! echo "$HEALTH" | grep -q '"model_loaded":true'; then
    warn "Model not loaded yet — it may still be downloading. Check: docker logs -f gemma-api"
fi

# =============================================================================
# STEP 9 — Summary
# =============================================================================
section "Deployment complete"

PUBLIC_IP=$(curl -sf http://checkip.amazonaws.com 2>/dev/null || curl -sf http://ifconfig.me 2>/dev/null || echo "<your-server-ip>")

echo -e "
${BOLD}┌───────────────────────────────────────────────────────┐
│              GEMMA 4 API — LIVE                       │
├───────────────────────────────────────────────────────┤${RESET}
  Health check  : http://$PUBLIC_IP:$API_PORT/health
  Analyze       : POST http://$PUBLIC_IP:$API_PORT/analyze
  Generate      : POST http://$PUBLIC_IP:$API_PORT/generate
  Prometheus    : http://$PUBLIC_IP:$PROMETHEUS_PORT
  Grafana       : http://$PUBLIC_IP:$GRAFANA_PORT
  Logs          : docker logs -f gemma-api
  CSV output    : $DATA_MOUNT/analyze_results.csv
${BOLD}├───────────────────────────────────────────────────────┤${RESET}
  API_KEY       : $API_KEY
  Grafana login : admin / $GRAFANA_PASSWORD
${BOLD}└───────────────────────────────────────────────────────┘${RESET}

${YELLOW}Open these ports in your EC2 Security Group: $API_PORT, $PROMETHEUS_PORT, $GRAFANA_PORT${RESET}
${YELLOW}On reboot, everything restarts automatically via gemma-startup.service.${RESET}
"
