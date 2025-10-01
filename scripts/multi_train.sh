

#!/bin/bash


# usage: bash multi_train.sh --gpu <GPU> --seed <seed num> --type <baseline/contact/saferl>

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
		*)
		echo "unknown: $1"; exit 1
		;;
	esac
done

wandb_project_name="tracking_contact"

case "$type" in
	baseline)
		# armswing
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-Baseline-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/armswing_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name armswing_baseline_seed${seed} --seed $seed &
		# legswing
		sleep 5
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-Baseline-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/legswing_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name legswing_baseline_seed${seed} --seed $seed &
		# halfsquat
		sleep 5
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-Baseline-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/halfsquat_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name halfsquat_baseline_seed${seed} --seed $seed &
		;;
	contact)
		# armswing
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-Contact-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/armswing_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name armswing_contact_seed${seed} --seed $seed &
		# legswing
		sleep 5
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-Contact-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/legswing_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name legswing_contact_seed${seed} --seed $seed &
		# halfsquat
		sleep 5
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-Contact-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/halfsquat_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name halfsquat_contact_seed${seed} --seed $seed &
		;;
	saferl)
		# armswing
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-SafeRL-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/armswing_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name armswing_saferl_seed${seed} --seed $seed --saferl &
		# legswing
		sleep 5
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-SafeRL-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/legswing_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name legswing_saferl_seed${seed} --seed $seed --saferl &
		# halfsquat
		sleep 5
		CUDA_VISIBLE_DEVICES=$cuda_device python scripts/rsl_rl/train.py --task=Tracking-SafeRL-Flat-G1-v0 --registry_name seu-ai-org/wandb-registry-motions/halfsquat_withcontact --headless --logger wandb --log_project_name $wandb_project_name --run_name halfsquat_saferl_seed${seed} --seed $seed --saferl &
		;;
	*)
		echo "type must be baseline/contact/saferl"; exit 1
		;;
esac
