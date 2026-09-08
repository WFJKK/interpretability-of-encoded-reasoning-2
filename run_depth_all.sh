#!/usr/bin/env bash
# Launch the depth measurement for both base models in parallel; safe to rerun (resumes).
mkdir -p runs
nohup bash run_depth.sh gpt2 > runs/depth_gpt2.log 2>&1 &
sleep 5
nohup bash run_depth.sh Qwen/Qwen2.5-0.5B > runs/depth_qwen.log 2>&1 &
echo "launched; check with: tail -5 runs/depth_gpt2.log runs/depth_qwen.log"
