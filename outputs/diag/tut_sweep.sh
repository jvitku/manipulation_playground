#!/usr/bin/env bash
# TA tutorial sweep: CPU queue (4 parallel) now; GPU queue (1 at a time) once the SmolVLA pair is done.
set -u
cd /home/jvitku/workspace/manipulation_playground
( xargs -P 4 -L 1 outputs/diag/tut_run_one.sh cpu < outputs/diag/tut_jobs_cpu.txt ) &
( until [ -f outputs/ta_ta_aux/checkpoints/010000/pretrained_model/model.safetensors ] && [ -f outputs/ta_ta_s2/checkpoints/010000/pretrained_model/model.safetensors ]; do sleep 60; done
  sleep 120
  xargs -P 1 -L 1 outputs/diag/tut_run_one.sh gpu < outputs/diag/tut_jobs_gpu.txt ) &
wait
echo all done
