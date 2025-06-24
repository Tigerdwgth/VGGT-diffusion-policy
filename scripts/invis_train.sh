# bash scripts/train_policy.sh dp {task} {Train_name} 0 {GPUID} {eval_robovis} {robovis} {policy_prio_as_cond} {policy_visual_prio_training}
# search_space:
#   #Invisibòto
#   #env
#   eval_robovis: choice(true,false)
#   robovis: choice(true,false)
#   policy.prio_as_cond: choice(true,false)
#   policy.visual_prio_training: choice(true,false)
#   task: [
#     metaworld_peg-unplug-side,
#     metaworld_coffee-pull,
#     metaworld_push,
#     metaworld_disassemble,
#   ]
#gpu available: 0,1,2,3,4,5,6
eval_robovis_values=(true false)
robovis_values=(true false)
prio_as_cond_values=(true false)
visual_prio_training_values=(true false)
tasks=(
  "metaworld_peg-unplug-side"
  "metaworld_coffee-pull"
  "metaworld_push"
  "metaworld_disassemble"
)
#egl visible
bash scripts/train_policy.sh dp metaworld_peg-unplug-side dp-peg-unplug-side-evalvis1-robovis1-pac1-vpt1 0 0 true true true true
bash scripts/train_policy.sh dp metaworld_peg-unplug-side dp-peg-unplug-side-evalvis1-robovis1-pac1-vpt0 0 0 true true true false
bash scripts/train_policy.sh dp metaworld_peg-unplug-side dp-peg-unplug-side-evalvis1-robovis1-pac0-vpt1 0 1 true true false true
bash scripts/train_policy.sh dp metaworld_peg-unplug-side dp-peg-unplug-side-evalvis1-robovis1-pac0-vpt0 0 2 true true false false
bash scripts/train_policy.sh dp metaworld_peg-unplug-side dp-peg-unplug-side-evalvis0-robovis0-pac1-vpt1 0 3 false false true true
bash scripts/train_policy.sh dp metaworld_peg-unplug-side dp-peg-unplug-side-evalvis0-robovis0-pac1-vpt0 0 4 false false true false
bash scripts/train_policy.sh dp metaworld_peg-unplug-side dp-peg-unplug-side-evalvis0-robovis0-pac0-vpt1 0 5 false false false true
bash scripts/train_policy.sh dp metaworld_peg-unplug-side dp-peg-unplug-side-evalvis0-robovis0-pac0-vpt0 0 6 false false false false

bash scripts/train_policy.sh dp metaworld_coffee-pull dp-coffee-pull-evalvis1-robovis1-pac1-vpt1 0 0 true true true true
bash scripts/train_policy.sh dp metaworld_coffee-pull dp-coffee-pull-evalvis1-robovis1-pac1-vpt0 0 0 true true true false
bash scripts/train_policy.sh dp metaworld_coffee-pull dp-coffee-pull-evalvis1-robovis1-pac0-vpt1 0 1 true true false true
bash scripts/train_policy.sh dp metaworld_coffee-pull dp-coffee-pull-evalvis1-robovis1-pac0-vpt0 0 2 true true false false
bash scripts/train_policy.sh dp metaworld_coffee-pull dp-coffee-pull-evalvis0-robovis0-pac1-vpt1 0 3 false false true true
bash scripts/train_policy.sh dp metaworld_coffee-pull dp-coffee-pull-evalvis0-robovis0-pac1-vpt0 0 4 false false true false
bash scripts/train_policy.sh dp metaworld_coffee-pull dp-coffee-pull-evalvis0-robovis0-pac0-vpt1 0 5 false false false true
bash scripts/train_policy.sh dp metaworld_coffee-pull dp-coffee-pull-evalvis0-robovis0-pac0-vpt0 0 6 false false false false

bash scripts/train_policy.sh dp metaworld_push dp-push-evalvis1-robovis1-pac1-vpt1 0 0 true true true true
bash scripts/train_policy.sh dp metaworld_push dp-push-evalvis1-robovis1-pac1-vpt0 0 0 true true true false
bash scripts/train_policy.sh dp metaworld_push dp-push-evalvis1-robovis1-pac0-vpt1 0 1 true true false true
bash scripts/train_policy.sh dp metaworld_push dp-push-evalvis1-robovis1-pac0-vpt0 0 2 true true false false
bash scripts/train_policy.sh dp metaworld_push dp-push-evalvis0-robovis0-pac1-vpt1 0 3 false false true true
bash scripts/train_policy.sh dp metaworld_push dp-push-evalvis0-robovis0-pac1-vpt0 0 4 false false true false
bash scripts/train_policy.sh dp metaworld_push dp-push-evalvis0-robovis0-pac0-vpt1 0 5 false false false true
bash scripts/train_policy.sh dp metaworld_push dp-push-evalvis0-robovis0-pac0-vpt0 0 6 false false false false

bash scripts/train_policy.sh dp metaworld_disassemble dp-disassemble-evalvis1-robovis1-pac1-vpt1 0 0 true true true true
bash scripts/train_policy.sh dp metaworld_disassemble dp-disassemble-evalvis1-robovis1-pac1-vpt0 0 0 true true true false
bash scripts/train_policy.sh dp metaworld_disassemble dp-disassemble-evalvis1-robovis1-pac0-vpt1 0 1 true true false true
bash scripts/train_policy.sh dp metaworld_disassemble dp-disassemble-evalvis1-robovis1-pac0-vpt0 0 2 true true false false
bash scripts/train_policy.sh dp metaworld_disassemble dp-disassemble-evalvis0-robovis0-pac1-vpt1 0 3 false false true true
bash scripts/train_policy.sh dp metaworld_disassemble dp-disassemble-evalvis0-robovis0-pac1-vpt0 0 4 false false true false
bash scripts/train_policy.sh dp metaworld_disassemble dp-disassemble-evalvis0-robovis0-pac0-vpt1 0 5 false false false true
bash scripts/train_policy.sh dp metaworld_disassemble dp-disassemble-evalvis0-robovis0-pac0-vpt0 0 6 false false false false


