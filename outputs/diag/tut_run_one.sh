#!/usr/bin/env bash
# one tutorial job; $1 = cpu|gpu, rest = args of scripts/40_ta_tutorial.py (skips finished runs)
set -u
cd /home/jvitku/workspace/manipulation_playground
mode=$1; shift
out=$(echo "$@" | sed 's/.*--out \([^ ]*\).*/\1/')
[ -f "$out/final.json" ] && exit 0
mkdir -p "$out"
env_flag=""; [ "$mode" = cpu ] && env_flag="-e CUDA_VISIBLE_DEVICES="
docker compose -f docker/compose.yaml run --rm -T -u "$(id -u):$(id -g)" $env_flag vla python scripts/40_ta_tutorial.py "$@" > "$out/run.log" 2>&1
echo "$(date +%H:%M) $out $(tail -1 "$out/run.log" | cut -c1-160)"
