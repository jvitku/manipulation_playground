#!/usr/bin/env bash
# Hourly progress summary: build docs/progress/index.html with the host's local time.
# Usage: docs/progress/make.sh "what happens next"
set -euo pipefail
cd "$(dirname "$0")/../.."
# the container runs in UTC; give it the host's current UTC offset as a POSIX TZ (sign inverted)
# so `git log --since=<local time>` counts the right commits
off=$(date +%z)
tz="LOC$( [ "${off:0:1}" = "+" ] && echo - || echo + )${off:1:2}:${off:3:2}"
docker compose -f docker/compose.yaml run --rm -T -u "$(id -u):$(id -g)" -e TZ="$tz" dev \
  python docs/progress/build.py --now "$(date +%Y-%m-%dT%H:%M)" --next "${1:-}"
