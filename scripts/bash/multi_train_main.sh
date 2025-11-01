#!/bin/bash

# usage: bash multi_train.sh --gpu <GPU> --seed <seed num> --type <baseline/contact/saferl> --task <task1,task2,...>

# args
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --gpu)
        cuda_device="$2"
        shift; shift
        ;;
        --seed)
        seed="$2"
        shift; shift
        ;;
        --type)
        type="$2"
        shift; shift
        ;;
        --task)
        tasks="$2"  
        shift; shift
        ;;
        *)
        echo "unknown: $1"; exit 1
        ;;
    esac
done

wandb_project_name="tracking_contact"
wandb_org="ziang_zheng-tsinghua-university-org"

IFS=',' read -r -a task_array <<< "$tasks"

for task in "${task_array[@]}"; do
    case "$type" in
        baseline)
            CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-Baseline-Flat-G1-v0 --registry_name $wandb_org/wandb-registry-motions/${task}_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name ${task}_baseline_seed${seed} --seed $seed &
            ;;
        penalty)
            CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-Contact-Flat-G1-v0 --registry_name $wandb_org/wandb-registry-motions/${task}_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name ${task}_contact_seed${seed} --seed $seed &
            ;;
        saferl)
            CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-SafeRL-Flat-G1-v0 --registry_name $wandb_org/wandb-registry-motions/${task}_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name ${task}_saferl_seed${seed} --seed $seed --saferl &
            ;;
        deploy)
            CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Deploy-Tracking-Flat-G1-v0 --registry_name $wandb_org/wandb-registry-motions/${task}_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name deploy_${task}_baseline_seed${seed} --seed $seed &
            ;;
        deploy_saferl)
            CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Deploy-Tracking-SafeRL-Flat-G1-v0 --registry_name $wandb_org/wandb-registry-motions/${task}_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name deploy_${task}_saferl_seed${seed} --seed $seed --saferl &
            ;;
        *)
            echo "type must be baseline/saferl"; exit 1
            ;;
    esac
	sleep 5
done