#!/usr/bin/env bash
# ============================================================
# ECS Server Setup Script — ETF Dashboard
# Run once after SSH-ing into a fresh Ubuntu/Debian ECS instance.
# Usage:  bash setup-server.sh
# ============================================================
set -euo pipefail

APP_DIR="/opt/etf-dashboard"
REPO_URL=""  # <-- Fill in your git repo URL, or scp the code manually

echo "=== 1. System packages ==="
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv git

echo "=== 2. Create app directory ==="
sudo mkdir -p "$APP_DIR"
sudo chown "$(whoami):$(whoami)" "$APP_DIR"

# --- Option A: Clone from git (fill in REPO_URL above) ---
if [ -n "$REPO_URL" ]; then
    git clone "$REPO_URL" "$APP_DIR"
fi
# --- Option B: If you scp-ed the code, skip the clone ---

echo "=== 3. Python virtual environment ==="
cd "$APP_DIR"
python3 -m venv .venv
source .venv/bin/activate

echo "=== 4. Install dependencies (dashboard-only, lightweight) ==="
pip install --upgrade pip
pip install -r deploy/requirements-deploy.txt

echo "=== 5. Create .env (you MUST edit this) ==="
if [ ! -f .env ]; then
    cat > .env <<'ENVEOF'
ETF_DB_URL=postgresql://ETF_Dashboard:YOUR_PASSWORD_HERE@pgm-bp172lqhb76i9xyvdo.pg.rds.aliyuncs.com:5432/etf_strategy
ENVEOF
    echo ">>> IMPORTANT: edit $APP_DIR/.env and set the real password <<<"
fi

echo "=== 6. Install systemd service ==="
sudo cp deploy/etf-dashboard.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable etf-dashboard
sudo systemctl start etf-dashboard

echo "=== Done! ==="
echo "Dashboard is running on port 8501."
echo "Check status:  sudo systemctl status etf-dashboard"
echo "View logs:     sudo journalctl -u etf-dashboard -f"
