#!/usr/bin/env bash
# V9: TA with z-scored torque history; trains after the plain run, then the same 100-set-up eval.
set -u
cd /home/jvitku/workspace/manipulation_playground
U="$(id -u):$(id -g)"
DC="docker compose -f docker/compose.yaml"
REN=observation.images.front=observation.images.camera1,observation.images.wrist=observation.images.camera2
until [ -f outputs/ta_off/checkpoints/010000/pretrained_model/model.safetensors ] && ! docker ps --format '{{.Names}}' | grep -qx tatrain_off; do sleep 60; done
rm -rf outputs/ta_ta_norm
$DC run --rm -T -u $U -e TA_MODE=ta_norm -e TA_STATS=outputs/lerobot/insert_ta/meta/stats.json --name tatrain_ta_norm vla python scripts/34_train_ta_smolvla.py --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda --dataset.repo_id=local/insert_ta --dataset.root=outputs/lerobot/insert_ta --dataset.video_backend=pyav '--rename_map={"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --policy.empty_cameras=1 --output_dir=outputs/ta_ta_norm --job_name=ta_ta_norm --batch_size=8 --steps=10000 --save_freq=5000 --log_freq=200 --num_workers=3 --wandb.enable=false > outputs/ta_ta_norm.log 2>&1
ck=outputs/ta_ta_norm/checkpoints/010000/pretrained_model
$DC run --rm -T -u $U --name srv_ta_norm vla python scripts/24_serve_policy.py --ckpt $ck --ta ta_norm --n-action-steps 10 --rename $REN > outputs/ta_srv_ta_norm.log 2>&1 &
until grep -q "serving\|Error" outputs/ta_srv_ta_norm.log 2>/dev/null; do sleep 10; done
rm -rf outputs/eval_ta_insert/ta_norm
$DC run --rm -T -u $U dev python scripts/35_eval_insert_skill.py --host srv_ta_norm --seeds 0-99 --out outputs/eval_ta_insert/ta_norm --timeout 120 > outputs/eval_ta_insert_ta_norm.log 2>&1
docker stop srv_ta_norm >/dev/null
echo done
