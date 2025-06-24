# bash scripts/gen_demonstration_metaworld.sh basketball
# bash scripts/gen_demonstration_metaworld.sh pick-out-of-hole

# Assembly Hand Insert Pick Out of Hole Pick Place
# bash scripts/gen_demonstration_metaworld.sh pick-out-of-hole
# bash scripts/gen_demonstration_metaworld.sh pick-place 
# bash scripts/gen_demonstration_metaworld.sh hand-insert
# bash scripts/gen_demonstration_metaworld.sh assembly
# Reach Wall, Window Close, Window Open 
# bash scripts/gen_demonstration_metaworld.sh reach-wall
# bash scripts/gen_demonstration_metaworld.sh window-close 
# bash scripts/gen_demonstration_metaworld.sh window-close False
# bash scripts/gen_demonstration_metaworld.sh window-open
# Shelf Place
# bash scripts/gen_demonstration_metaworld.sh shelf-place

cd third_party/Metaworld

task_name=${1}
robovis=${2}
export CUDA_VISIBLE_DEVICES=0
python gen_demonstration_expert.py --env_name=${task_name} \
            --num_episodes 10 \
            --root_dir "../../3D-Diffusion-Policy/data/" \
            --robovis=${robovis} \
