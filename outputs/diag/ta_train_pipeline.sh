#!/usr/bin/env bash
set -u
cd /home/jvitku/workspace/manipulation_playground
U="$(id -u):$(id -g)"
run() {  # mode
  rm -rf outputs/ta_$1
  docker compose -f docker/compose.yaml run --rm -T -u $U -e TA_MODE=$1 --name tatrain_$1 vla python scripts/34_train_ta_smolvla.py --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --policy.device=cuda --dataset.repo_id=local/insert_ta --dataset.root=outputs/lerobot/insert_ta --dataset.video_backend=pyav '--rename_map={"observation.images.front": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}' --policy.empty_cameras=1 --output_dir=outputs/ta_$1 --job_name=ta_$1 --batch_size=8 --steps=10000 --save_freq=5000 --log_freq=200 --num_workers=3 --wandb.enable=false > outputs/ta_$1.log 2>&1
}
run ta & sleep 90; run ta_zero & wait
run off
echo done
