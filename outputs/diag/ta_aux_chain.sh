#!/usr/bin/env bash
# V9 follow-up: TA + auxiliary torque prediction (ta_aux), and a seed-2 replicate of the raw TA arm.
set -u
cd /home/jvitku/workspace/manipulation_playground
U="$(id -u):$(id -g)"
DC="docker compose -f docker/compose.yaml"
REN=observation.images.front=observation.images.camera1,observation.images.wrist=observation.images.camera2
arm() {  # name mode dataset seed
  local n=$1 m=$2 d=$3 s=$4
  rm -rf outputs/ta_$n
  $DC run --rm -T -u $U -e TA_MODE=$m -e TA_STATS=outputs/lerobot/$d/meta/stats.json --name tatrain_$n vla python scripts/34_train_ta_smolvla.py --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda --dataset.repo_id=local/$d --dataset.root=outputs/lerobot/$d --dataset.video_backend=pyav '--rename_map={"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --policy.empty_cameras=1 --output_dir=outputs/ta_$n --job_name=ta_$n --seed=$s --batch_size=8 --steps=10000 --save_freq=5000 --log_freq=200 --num_workers=3 --wandb.enable=false > outputs/ta_$n.log 2>&1
  $DC run --rm -T -u $U --name srv_$n vla python scripts/24_serve_policy.py --ckpt outputs/ta_$n/checkpoints/010000/pretrained_model --ta $m --n-action-steps 10 --rename $REN > outputs/ta_srv_$n.log 2>&1 &
  until grep -q "serving\|Error" outputs/ta_srv_$n.log 2>/dev/null; do sleep 10; done
  rm -rf outputs/eval_ta_insert/$n
  $DC run --rm -T -u $U dev python scripts/35_eval_insert_skill.py --host srv_$n --seeds 0-99 --out outputs/eval_ta_insert/$n --timeout 120 > outputs/eval_ta_insert_$n.log 2>&1
  docker stop srv_$n >/dev/null
}
arm ta_aux ta_aux insert_ta_fut 1000 &
sleep 120
arm ta_s2 ta insert_ta 2 &
wait
echo done
