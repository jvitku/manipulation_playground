#!/usr/bin/env bash
# Hourly progress summary: build docs/progress/index.html with the host's local time.
# Usage: docs/progress/make.sh "what happens next"
set -euo pipefail
cd "$(dirname "$0")/../.."
docker compose -f docker/compose.yaml run --rm -T -u "$(id -u):$(id -g)" dev \
  python docs/progress/build.py --now "$(date +%Y-%m-%dT%H:%M)" --next "${1:-}"
