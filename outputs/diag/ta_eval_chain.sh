#!/usr/bin/env bash
# V9: serve each TA-SmolVLA fine-tune (final checkpoint) and evaluate it on 100 held-out insertion set-ups.
set -u
cd /home/jvitku/workspace/manipulation_playground
U="$(id -u):$(id -g)"
DC="docker compose -f docker/compose.yaml"
REN=observation.images.front=observation.images.camera1,observation.images.wrist=observation.images.camera2
evalone() {  # mode
  local m=$1 ck=outputs/ta_$1/checkpoints/010000/pretrained_model
  until [ -f $ck/model.safetensors ] && ! docker ps --format '{{.Names}}' | grep -qx tatrain_$m; do sleep 60; done
  $DC run --rm -T -u $U --name srv_$m vla python scripts/24_serve_policy.py --ckpt $ck --ta $m --n-action-steps 10 --rename $REN > outputs/ta_srv_$m.log 2>&1 &
  until grep -q "serving\|Error" outputs/ta_srv_$m.log 2>/dev/null; do sleep 10; done
  rm -rf outputs/eval_ta_insert/$m
  $DC run --rm -T -u $U dev python scripts/35_eval_insert_skill.py --host srv_$m --seeds 0-99 --out outputs/eval_ta_insert/$m --timeout 120 > outputs/eval_ta_insert_$m.log 2>&1
  docker stop srv_$m >/dev/null
}
mkdir -p outputs/eval_ta_insert
evalone ta & evalone ta_zero & wait
evalone off
echo done
