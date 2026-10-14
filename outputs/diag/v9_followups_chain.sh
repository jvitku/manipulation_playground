#!/usr/bin/env bash
# V9 follow-ups (2026-10-14): rerun of the crashed ta_aux / ta seed-2 arms, a seed-2 zeroed twin,
# and the ACT +/- force-token arm (2 seeds each). Two GPU lanes; every arm is evaluated on the same
# 100 held-out insertion set-ups (scripts/35_eval_insert_skill.py).
set -u
cd /home/jvitku/workspace/manipulation_playground
U="$(id -u):$(id -g)"
DC="docker compose -f docker/compose.yaml"
SV=observation.images.front=observation.images.camera1,observation.images.wrist=observation.images.camera2
evalarm() {  # name ckpt server-args...
  local n=$1 ck=$2; shift 2
  $DC run --rm -T -u $U --name srv_$n vla python scripts/24_serve_policy.py --ckpt $ck --n-action-steps 10 "$@" > outputs/srv_$n.log 2>&1 &
  until grep -q "serving\|Error" outputs/srv_$n.log 2>/dev/null; do sleep 10; done
  rm -rf outputs/eval_ta_insert/$n
  $DC run --rm -T -u $U dev python scripts/35_eval_insert_skill.py --host srv_$n --seeds 0-99 --out outputs/eval_ta_insert/$n --timeout 120 > outputs/eval_ta_insert_$n.log 2>&1
  docker stop srv_$n >/dev/null
}
smol() {  # name mode dataset seed
  local n=$1 m=$2 d=$3 s=$4
  rm -rf outputs/ta_$n
  $DC run --rm -T -u $U -e TA_MODE=$m -e TA_STATS=outputs/lerobot/$d/meta/stats.json --name tatrain_$n vla python scripts/34_train_ta_smolvla.py --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda --dataset.repo_id=local/$d --dataset.root=outputs/lerobot/$d --dataset.video_backend=pyav '--rename_map={"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --policy.empty_cameras=1 --output_dir=outputs/ta_$n --job_name=ta_$n --seed=$s --batch_size=8 --steps=10000 --save_freq=5000 --log_freq=200 --num_workers=3 --wandb.enable=false > outputs/ta_$n.log 2>&1
  evalarm $n outputs/ta_$n/checkpoints/010000/pretrained_model --ta $m --rename $SV
}
act() {  # name dataset seed server-rename
  local n=$1 d=$2 s=$3 r=$4
  rm -rf outputs/act_$n
  $DC run --rm -T -u $U --name acttrain_$n vla lerobot-train --policy.type=act --policy.push_to_hub=false --policy.device=cuda --dataset.repo_id=local/$d --dataset.root=outputs/lerobot/$d --dataset.video_backend=pyav --output_dir=outputs/act_$n --job_name=act_$n --seed=$s --batch_size=16 --steps=20000 --save_freq=10000 --log_freq=200 --num_workers=3 --wandb.enable=false > outputs/act_$n.log 2>&1
  evalarm act_$n outputs/act_$n/checkpoints/020000/pretrained_model ${r:+--rename $r}
}
lane1() { smol ta_aux ta_aux insert_ta_fut 1000; smol ta_zero_s2 ta_zero insert_ta 2; }
lane2() {
  smol ta_s2 ta insert_ta 2
  act f_s0 insert_act_env 0 observation.torque_hist=observation.environment_state
  act nf_s0 insert_act_nf 0 ""
  act f_s1 insert_act_env 1 observation.torque_hist=observation.environment_state
  act nf_s1 insert_act_nf 1 ""
}
lane1 & sleep 120; lane2 & wait
echo done
