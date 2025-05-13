from typing import Dict
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import numpy as np
import pytorch3d.ops as torch3d_ops


from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import DP3Encoder
from diffusion_policy_3d.model.vision.pointnet_extractor import create_mlp

class DP(BasePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=256,
            crop_shape=None,
            use_pc_color=False,
            pointnet_type="pointnet",
            pointcloud_encoder_cfg=None,
            # parameters passed to step,
            pc_noise=False,
            prio_as_cond=True,
            visual_prio_training=True,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])


        # obs_encoder = DP3Encoder(observation_space=obs_dict,
        #                                            img_crop_shape=crop_shape,
        #                                         out_channel=encoder_output_dim,
        #                                         pointcloud_encoder_cfg=pointcloud_encoder_cfg,
        #                                         use_pc_color=use_pc_color,
        #                                         pointnet_type=pointnet_type,
        #                                         )
        #ues pretrained resnet18
        obs_encoder= DPEncoder(observation_space=obs_dict,
                                            img_crop_shape=crop_shape,
                                         out_channel=encoder_output_dim,
                                         pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                                         use_pc_color=use_pc_color,
                                         pointnet_type=pointnet_type,
                                         prio_as_cond=prio_as_cond,
                                         out_dim=encoder_output_dim
                                         )
            

        # create diffusion model
        obs_feature_dim = encoder_output_dim
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps
        print("n_obs_steps",n_obs_steps)
        print("condition_type",condition_type)
        print("global_cond_dim",global_cond_dim)        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DiffusionUnetHybridPointcloudPolicy] pointnet_type: {self.pointnet_type}", "yellow")



        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.prio_as_cond = prio_as_cond
        self.visual_prio_training = visual_prio_training
        
        #
        self.pc_noise=pc_noise
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        
        
        #本体感知模块，一个三层的mlp用于直接预测机器人的状态
        
        if self.visual_prio_training:
            self.proprioception_mlp = nn.Sequential(
            nn.Linear(obs_feature_dim, 512),
            nn.ReLU(), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 9),
        )
        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond, global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        # print(nobs.keys())
        # this_n_point_cloud = nobs['imagin_robot'][..., :3] # only use coordinate
        # if not self.use_pc_color:
        #     nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        # this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype
        

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        this_state=this_nobs['agent_pos']
        if self.obs_as_global_cond:
            # condition through global feature

            
            if this_nobs['image'].shape[1] != 3:
                this_nobs['image'] = this_nobs['image'].permute(0, 3, 1, 2)
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            
        else:
            # condition through impainting 
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            if this_nobs['image'].shape[1] != 3:
                this_nobs['image'] = this_nobs['image'].permute(0, 3, 1, 2)
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True


        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        #add some error to action_pred
        #error_rate=1
        #·error = (torch.rand_like(action) * 2 - 1) * error_rate * action
        # print(error,action)
        #action=action+error
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())


    def compute_loss(self, batch):

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        # normalize input
        # print("shape of batch pc",batch['obs']['point_cloud'].shape)
        # print("shape of batch image",batch['obs']['image'].shape)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            # print("shape of this_nobs",this_nobs.shape)
            # print(this_nobs.keys())
            if this_nobs['image'].shape[1] != 3:
                this_nobs['image'] = this_nobs['image'].permute(0, 3, 1, 2)
            nobs_features = self.obs_encoder(this_nobs)
            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
            # this_n_point_cloud = this_nobs['imagin_robot'].reshape(batch_size,-1, *this_nobs['imagin_robot'].shape[1:])
            # this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            # this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            # # print("shape of this_nobs",this_nobs)
            # print(this_nobs.keys())
            if this_nobs['image'].shape[1] != 3:
                this_nobs['image'] = this_nobs['image'].permute(0, 3, 1, 2)
            nobs_features = self.obs_encoder(this_nobs)

            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            if self.prio_as_cond:
                # print("shape of nobs_features",nobs_features.shape)
                cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        

        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # print("shape of noisy_trajectory",noisy_trajectory.shape)
        # print("shape of timesteps",timesteps.shape)
        # # print("shape of local_cond",local_cond.shape)
        print("this_nobs_features",this_nobs['agent_pos'].shape)
        print("shape of global_cond",global_cond.shape)
        print("self.prio as cond",self.prio_as_cond)

        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            global_cond=global_cond)
        

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        
        
        
        loss = loss.mean()
        #add proprioception loss
        
        # 本体感知
        # Predict the noise residual
        # print("shape of nobs_features",nobs_features.shape)
        # print("shape of this_nobs",this_nobs['agent_pos'].shape)
        # print("shape of actions",nactions.shape)
        if self.visual_prio_training:
            proprioception= self.proprioception_mlp(nobs_features)
            loss_proprioception = F.mse_loss(proprioception, this_nobs['agent_pos'], reduction='none')
            loss+= loss_proprioception.mean()
        

        loss_dict = {
                'bc_loss': loss.item(),
            }

        # print(f"t2-t1: {t2-t1:.3f}")
        # print(f"t3-t2: {t3-t2:.3f}")
        # print(f"t4-t3: {t4-t3:.3f}")
        # print(f"t5-t4: {t5-t4:.3f}")
        # print(f"t6-t5: {t6-t5:.3f}")
        
        return loss, loss_dict
    
class DPEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 prio_as_cond=True,
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        self.prio_as_cond = prio_as_cond
        
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        # print("encoder_output_dim",out_channel)
        cprint(f"[DP3Encoder] encoder_output_dim: {out_channel}", "yellow")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        self.extractor=torchvision.models.resnet18(pretrained=True)
        self.extractor.fc = nn.Linear(self.extractor.fc.in_features, out_dim)
        


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]
        
        
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DPEncoder] output dim: {self.n_output_channels}", "red")

    def _register_hook(self, module):
        def hook(module, input, output):
            if isinstance(output, tuple):
                self.feature = output[0].detach()
            else:
                self.feature = output.detach()
        module.register_forward_hook(hook)
        return module
    
    def get_feature(self):
        return self.feature
    
    def forward(self, observations: Dict) -> torch.Tensor:
        images = observations[self.rgb_image_key]
        # b*c*h*w
        # assert len(images.shape) == 4, cprint(f"point cloud shape: {images.shape}, length should be 4", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :images.shape[-1]] # align the last dim
            images = torch.concat([images, img_points], dim=1)
        
        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        # print("DP3extractor",{points.shape})
        pn_feat = self.extractor(images)    # B * out_channel
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        if self.prio_as_cond:
            
            final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels
