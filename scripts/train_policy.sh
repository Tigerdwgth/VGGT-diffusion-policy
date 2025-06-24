# Examples:
# bash scripts/train_policy.sh dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 dexart_laptop 0322 0 0
# bash scripts/train_policy.sh simple_dp3 adroit_hammer 0322 0 0
# bash scripts/train_policy.sh dp3 metaworld_basketball 0602 0 0
# bash scripts/train_policy.sh dp metaworld_window-close 0602 0 0



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name=${alg_name}
addition_info=${3}
seed=${4}
eval_robovis=${6}
robovis=${7}
policy_prio_as_cond=${8}
policy_visual_prio_training=${9}

#默认值
if [ -z ${eval_robovis} ]; then
    eval_robovis=true
fi
if [ -z ${robovis} ]; then
    robovis=true
fi
if [ -z ${policy_prio_as_cond} ]; then
    policy_prio_as_cond=true
fi
if [ -z ${policy_visual_prio_training} ]; then
    policy_visual_prio_training=false
fi

exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


# gpu_id=$(bash scripts/find_gpu.sh)
gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi

cd 3D-Diffusion-Policy


export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
python train.py --config-name=${config_name}.yaml \
                            task=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt}\
                            eval_robovis=${eval_robovis} \
                            robovis=${robovis} \
                            policy.prio_as_cond=${policy_prio_as_cond} \
                            policy.visual_prio_training=${policy_visual_prio_training} \



                                