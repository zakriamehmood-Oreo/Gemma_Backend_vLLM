#!/bin/bash
# =============================================================================
# Gemma 4 API — Startup Script
# Runs on boot via systemd (gemma-startup.service) or manually: bash startup.sh
# =============================================================================

set -e

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; }

APP_DIR="/data/Gemma_Backend_vLLM"
VENV="/data/gemma-env"
HF_CACHE="/data/hf_cache"
LOG_FILE="/data/api.log"
DATA_MOUNT="/data"
DATA_UUID="e694f083-aac2-4981-9cb4-5acfe61bfe30"

# =============================================================================
# STEP 1 — Mount /data if not already mounted
# =============================================================================
info "Checking /data mount..."

if mountpoint -q "$DATA_MOUNT"; then
    success "/data already mounted"
else
    warn "/data not mounted — mounting now"
    sudo mount UUID="$DATA_UUID" "$DATA_MOUNT" -t ext4 -o defaults,nofail
    if mountpoint -q "$DATA_MOUNT"; then
        success "/data mounted ($(df -h $DATA_MOUNT | awk 'NR==2{print $4}') free)"
    else
        error "/data mount failed — aborting"
        exit 1
    fi
fi

# =============================================================================
# STEP 2 — Restore containerd symlink if missing (survives reboots via fstab
#           but the symlink target needs to exist first)
# =============================================================================
if [ ! -L /var/lib/containerd ] && [ -d "$DATA_MOUNT/containerd" ]; then
    info "Restoring containerd symlink"
    sudo rm -rf /var/lib/containerd
    sudo ln -s "$DATA_MOUNT/containerd" /var/lib/containerd
    sudo systemctl restart containerd
    success "containerd symlink restored"
fi

# =============================================================================
# STEP 3 — Wait for Docker
# =============================================================================
info "Waiting for Docker..."
for i in $(seq 1 15); do
    if sudo docker info > /dev/null 2>&1; then
        success "Docker is ready"
        break
    fi
    [ "$i" -eq 15 ] && { error "Docker did not start in time"; exit 1; }
    sleep 2
done

# =============================================================================
# STEP 4 — Load NVIDIA kernel module (if driver installed)
# =============================================================================
if modinfo nvidia > /dev/null 2>&1; then
    if ! lsmod | grep -q '^nvidia '; then
        info "Loading NVIDIA kernel module"
        sudo modprobe nvidia && sudo modprobe nvidia-uvm && sudo modprobe nvidia-modeset
        success "NVIDIA module loaded"
    else
        success "NVIDIA module already loaded"
    fi
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null && true
    sudo nvidia-smi -pm 1
    sudo nvidia-smi --lock-gpu-clocks=1590,1590
    success "GPU persistence mode enabled, clocks locked at 1590 MHz"
else
    warn "NVIDIA driver not found — running on CPU"
fi

# =============================================================================
# STEP 5 — Start monitoring stack
# =============================================================================
info "Starting monitoring stack (Prometheus, Grafana, Loki, Node Exporter)..."
sudo docker compose -f "$APP_DIR/docker-compose.monitoring.yml" up -d
success "Monitoring stack started"

# =============================================================================
# STEP 6 — Start Gemma API container
# =============================================================================
info "Starting Gemma API container..."

# Remove any stopped gemma-api container so docker run won't conflict
sudo docker rm -f gemma-api 2>/dev/null || true

sudo docker run -d \
    --name gemma-api \
    --runtime=nvidia \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -p 8000:8000 \
    --env-file "$APP_DIR/.env" \
    -v "$HF_CACHE:$HF_CACHE" \
    -v /data:/data \
    -v "$APP_DIR/app:/app/app" \
    --restart unless-stopped \
    gemma-api:latest \
    sh -c 'uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir /app/app 2>&1 | tee -a /data/api.log'

success "Gemma API container started"

# =============================================================================
# STEP 7 — Health check
# =============================================================================
info "Waiting for API to be ready..."
for i in $(seq 1 20); do
    if curl -s http://localhost:8000/health | grep -q '"status":"ok"'; then
        success "API is healthy"
        break
    fi
    [ "$i" -eq 20 ] && { warn "API did not respond in time — check: tail -f $LOG_FILE"; }
    sleep 5
done

# =============================================================================
# STEP 8 — Summary
# =============================================================================
echo -e "
${BOLD}┌──────────────────────────────────────────┐
│         All services running             │
├──────────────────────────────────────────┤${RESET}
  Gemma API   → http://localhost:8000/health
  Prometheus  → http://localhost:9090
  Grafana     → http://localhost:3000
  Loki        → http://localhost:3100/ready
  API logs    → tail -f $LOG_FILE
${BOLD}└──────────────────────────────────────────┘${RESET}
"
