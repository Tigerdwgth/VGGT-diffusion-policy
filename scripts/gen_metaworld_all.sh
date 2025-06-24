# 生成所有任务的隐形和不隐形示范

cd third_party/Metaworld

tasks=(
  "peg-unplug-side"
  "coffee-pull"
  "push"
  "disassemble"
)


export CUDA_VISIBLE_DEVICES=0

for task_name in "${tasks[@]}"; do
    # 生成不隐形的示范
    python gen_demonstration_expert.py --env_name=${task_name} \
        --num_episodes 15 \
        --root_dir "../../3D-Diffusion-Policy/data/" \
        --robovis=True

    # 生成隐形的示范
    python gen_demonstration_expert.py --env_name=${task_name} \
        --num_episodes 15 \
        --root_dir "../../3D-Diffusion-Policy/data/" \
        --robovis=False
done