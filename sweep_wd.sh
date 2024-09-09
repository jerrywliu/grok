#!/bin/bash

# split_ratios=$(seq 0.1 0.05 0.95)
split_ratios=$(seq 95 -5 10)
gpus=(0 1 2 3)

# Weight decay values to sweep
weight_decays=(0.000001 0.00001 0.0001 0.001)

# Sweep args
math_operator="x**2+y**2+x*y+x_mod_97"
random_seed=5
max_steps=500000

cmd="./scripts/train.py --random_seed ${random_seed} --math_operator ${math_operator}"
logdir="/scratch/grok/sweep_logs/${math_operator}_seed${random_seed}"
mkdir -p ${logdir}

# Start jobs
for wd in "${weight_decays[@]}"; do
    for ratio in $split_ratios; do
        while true; do
            for gpu in "${gpus[@]}"; do
                if ! screen -list | grep -q "gpu${gpu}"; then
                    # Make sure the log directory exists
                    mkdir -p ${logdir}/${ratio}_wd${wd}
                    full_cmd="source $(conda info --base)/etc/profile.d/conda.sh && conda activate grok-openai && ${cmd} --gpu ${gpu} --train_data_pct ${ratio} --weight_decay ${wd} --logdir ${logdir}_wd${wd}/${ratio} --max_steps ${max_steps} && exit"
                    screen -dmS "gpu${gpu}_task_${ratio}_wd${wd}" bash -c "${full_cmd}"
                    echo "Started job with split_ratio ${ratio}, weight_decay ${wd} on GPU ${gpu} in screen session gpu${gpu}_task_${ratio}_wd${wd}"
                    sleep 1  # Small delay to avoid potential race conditions
                    break 3  # Exit both loops and continue with the next split_ratio and weight_decay
                fi
            done
            sleep 5  # Wait for a short period before checking again
        done
    done
done

# Wait for all screens to finish before exiting the script
while screen -list | grep -q "gpu"; do
    sleep 10
done

echo "All jobs completed."
