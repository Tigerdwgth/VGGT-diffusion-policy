


# VGGDP



**Visual Geometry Grounded Diffusion Policy (VGGT DP)** is a visual imitation learning algorithm that utilize VGGT encoder as prior to learn a generalizable visuomotor policy. This repo is built upon the DP3 and VGGT.





# 📊 Benchmarks

**Simulation environments.** 
  currently, we support the `MetaWorld` for train and evaluation. Also We support metaworld evaluation with ood camera viewpoints.


**Algorithms**. We provide the implemntation of the following algorithms: 
- VGGT_dp:`vggt_dp.yaml`
- DP:`dp.yaml`
- DP3: `dp3.yaml` origin dp3 implementation
- Simple DP3: `simple_dp3.yaml`


Among these, `vggt_dp.yaml` is the proposed algorithm in our paper, showing a significant improvement over the baselines. During training, vggt_dp takes ~20G gpu memory and ~24 hours on an Nvidia L20 gpu.


# 💻 Installation

See [INSTALL.md](INSTALL.md) for installation instructions. 

See [ERROR_CATCH.md](ERROR_CATCH.md) for error catching I personally encountered during installation.

# 📚 Data
You could generate demonstrations by yourself using our provided expert policies.  Generated demonstrations are under `$YOUR_REPO_PATH/VGGDP/data/`.
- Download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and put the `ckpts` folder under `$YOUR_REPO_PATH/third_party/VRL3/`.
- Download DexArt assets from [Google Drive](https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing) and put the `assets` folder under `$YOUR_REPO_PATH/third_party/dexart-release/`.


**Note**: since you are generating demonstrations by yourselves, the results could be slightly different from the results reported in the paper. This is normal since the results of imitation learning highly depend on the demonstration quality. **Please re-generate demonstrations if you encounter some bad demonstrations** and **no need to open a new issue**.

# 🛠️ Usage
Scripts for generating demonstrations, training, and evaluation are all provided in the `scripts/` folder. 

The results are logged by `wandb`, so you need to `wandb login` first to see the results and videos.

For more detailed arguments, please refer to the scripts and the code. We here provide a simple instruction for using the codebase.

1. Generate demonstrations by `gen_demonstration_adroit.sh` and `gen_demonstration_dexart.sh`. See the scripts for details. For example:
    ```bash
    bash scripts/gen_demonstration_adroi
    ```
    This will generate demonstrations for the `hammer` task in Adroit environment. The data will be saved in `VGGDP/data/` folder automatically.


2. Train and evaluate a policy with behavior cloning. For example:
    ```bash
    bash scripts/train_policy.sh dp3 adroit_hammer 0112 0 0
    ```
    This will train a DP3 policy on the `hammer` task in Adroit environment using point cloud modality. By default we **save** the ckpt (optional in the script).


3. Evaluate a saved policy or use it for inference. Please set  For example:
    ```bash
    bash scripts/eval_policy.sh dp3 adroit_hammer 0112 0 0
    ```
    This will evaluate the saved DP3 policy you just trained. **Note: the evaluation script is only provided for deployment/inference. For benchmarking, please use the results logged in wandb during training.**
