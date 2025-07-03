Adroit
baseline dp3
bash scripts/train_policy.sh dp3 adroit_hammer baseline_dp_hammer 0 2
bash scripts/train_policy.sh dp3 adroit_door baseline_dp_door 0 3
bash scripts/train_policy.sh dp3 adroit_pen baseline_dp_pen 0 4

eval
bash scripts/eval_policy.sh dp3 adroit_hammer baseline_dp_hammer 0 0
bash scripts/eval_policy.sh dp3 adroit_door baseline_dp_door 0 0
bash scripts/eval_policy.sh dp3 adroit_pen baseline_dp_pen 0 0
dp3 dgcnn

bash scripts/train_policy.sh dp3_dgcnn adroit_hammer dp_dgcnn_hammer 0 0
bash scripts/train_policy.sh dp3_dgcnn adroit_door dp_dgcnn_door 0 1
bash scripts/train_policy.sh dp3_dgcnn adroit_pen dp_dgcnn_pen 0 0

lr1e-2
bash scripts/train_policy.sh dp3_dgcnn_lr1e-2 adroit_hammer dp_dgcnn_hammer_lr1e-2 0 0

Metaworld
baseline dp3
bash scripts/train_policy.sh dp3 metaworld_pick-place baseline_dp_pick-place 0 5
bash scripts/train_policy.sh dp3 metaworld_shelf-place baseline_dp_shelf-place 0 5
<!-- #本体感知 -->
<!-- bash line shelf-place -->
bash scripts/train_policy.sh dp3 metaworld_shelf-place proprioception_notshake_dp_shelf-place 0 4
bash scripts/train_policy.sh dp metaworld_pick-place proprioception_notshake_dp_pick-place 0 5

bash scripts/train_policy.sh dp3 metaworld_hand-insert proprioception_notshake_dp_hand-insert 0 3

bash scripts/train_policy.sh dp metaworld_assembly baseline_dp2_assembly 0 3
bash scripts/train_policy.sh dp3 metaworld_reach-wall baseline_dp_reach-wall 0 0
bash scripts/train_policy.sh dp3 metaworld_window-close baseline_dp_window-close 0 1
bash scripts/train_policy.sh dp3 metaworld_window-open baseline_dp_window-open 0 2 

bash scripts/eval_policy.sh dp3 metaworld_window-close baseline_dp_window-close 0 5
dp3 dgcnn
bash scripts/train_policy.sh dp3_dgcnn metaworld_pick-place dp_dgcnn_pick-place 0 0
bash scripts/train_policy.sh dp3_dgcnn metaworld_hand-insert dp_dgcnn_hand-insert 0 1
bash scripts/train_policy.sh dp3_dgcnn metaworld_assembly dp_dgcnn_assembly 0 0

sh scripts/train_policy.sh vggtq_dp metaworld_reach vggtqdp_reach 0 3 1 1 1 1