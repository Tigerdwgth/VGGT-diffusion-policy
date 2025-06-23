import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import os
import metaworld
import random
import time

from scipy.spatial.transform import Rotation as R
from natsort import natsorted
from termcolor import cprint
from gym import spaces
from diffusion_policy_3d.gym_util.mujoco_point_cloud import PointCloudGenerator
from diffusion_policy_3d.gym_util.mjpc_wrapper import point_cloud_sampling

TASK_BOUDNS = {
    'default': [-0.5, -1.5, -0.795, 1, -0.4, 100],
}

class MetaWorldEnv(gym.Env):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self, task_name, device="cuda:0", 
                 use_point_crop=True,
                 num_points=1024,
                 ):
        super(MetaWorldEnv, self).__init__()

        if '-v2' not in task_name:
            task_name = task_name + '-v2-goal-observable'

        self.env = metaworld.envs.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[task_name]()
        self.env._freeze_rand_vec = False

        # https://arxiv.org/abs/2212.05698
        # self.env.sim.model.cam_pos[2] = [0.75, 0.075, 0.7]
        print("model cameras: ", self.env.sim.model.cam_pos)
        print("model cameras: ", self.env.sim.model.cam_quat)
        print("camera counter: ", len(self.env.sim.model.cam_pos))
        # self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        self.env.sim.model.cam_pos[2] = [0.6, 0.295, 0.8]
        self.agent_poses=[]
        self.pcs=[]
        self.env.sim.model.vis.map.znear = 0.1
        self.env.sim.model.vis.map.zfar = 1.5
        
        self.device_id = int(device.split(":")[-1])
        
        self.image_size = 128
        
        self.pc_generator = PointCloudGenerator(sim=self.env.sim, cam_names=['corner2'], img_size=self.image_size)
        self.use_point_crop = use_point_crop
        cprint("[MetaWorldEnv] use_point_crop: {}".format(self.use_point_crop), "cyan")
        self.num_points = num_points # 512
        
        print(self.env.sim.model.cam_quat[2])
        self.origin_quat= self.env.sim.model.cam_quat[2].copy()
        #quaternion to euler
        eular= R.from_quat([self.env.sim.model.cam_quat[2][1], self.env.sim.model.cam_quat[2][2], self.env.sim.model.cam_quat[2][3], self.env.sim.model.cam_quat[2][0]]).as_euler('xyz', degrees=True)
        print("camera euler: ", eular)
        x_angle = 61.4
        y_angle = -7
        self.pc_transform = np.array([
            [1, 0, 0],
            [0, np.cos(np.deg2rad(x_angle)), np.sin(np.deg2rad(x_angle))],
            [0, -np.sin(np.deg2rad(x_angle)), np.cos(np.deg2rad(x_angle))]
        ]) @ np.array([
            [np.cos(np.deg2rad(y_angle)), 0, np.sin(np.deg2rad(y_angle))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(y_angle)), 0, np.cos(np.deg2rad(y_angle))]
        ])
        self.old_pc_transform = self.pc_transform.copy()
        self.pc_scale = np.array([1, 1, 1])
        self.pc_offset = np.array([0, 0, 0])
        if task_name in TASK_BOUDNS:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS[task_name]
        else:
            x_min, y_min, z_min, x_max, y_max, z_max = TASK_BOUDNS['default']
        self.min_bound = [x_min, y_min, z_min]
        self.max_bound = [x_max, y_max, z_max]
        
    
        self.episode_length = self._max_episode_steps = 200
        self.action_space = self.env.action_space
        self.obs_sensor_dim = self.get_robot_state().shape[0]

        
    
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            ),
            'depth': spaces.Box(
                low=0,
                high=255,
                shape=(self.image_size, self.image_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.obs_sensor_dim,),
                dtype=np.float32
            ),
            'point_cloud': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.num_points, 3),
                dtype=np.float32
            ),
            'full_state': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(20, ),
                dtype=np.float32
            ),
        })

    def get_robot_state(self):
        # print("robot state: ", self.env.get_endeff_pos())
        # print("reef pos: ", self.env._get_site_pos('rightEndEffector'))
        # print("finger pos: ", self.env._get_site_pos('leftEndEffector'))
        eef_pos = self.env.get_endeff_pos()
        finger_right, finger_left = (
            self.env._get_site_pos('rightEndEffector'),
            self.env._get_site_pos('leftEndEffector')
        )
        return np.concatenate([eef_pos, finger_right, finger_left])

    def get_rgb(self):
        
        img = self.env.sim.render(width=self.image_size, height=self.image_size, camera_name="corner2", device_id=self.device_id)
        
        return img

    def render_high_res(self, resolution=1024):
        img = self.env.sim.render(width=resolution, height=resolution, camera_name="corner2", device_id=self.device_id)
        return img
    

    def get_point_cloud(self, use_rgb=True):
        point_cloud, depth = self.pc_generator.generateCroppedPointCloud(device_id=self.device_id) # raw point cloud, Nx3
        
        
        if not use_rgb:
            point_cloud = point_cloud[..., :3]
        
        
        if self.pc_transform is not None:
            point_cloud[:, :3] = point_cloud[:, :3] @ self.pc_transform.T
        if self.pc_scale is not None:
            point_cloud[:, :3] = point_cloud[:, :3] * self.pc_scale
        
        if self.pc_offset is not None:    
            point_cloud[:, :3] = point_cloud[:, :3] + self.pc_offset
        
        if self.use_point_crop:
            if self.min_bound is not None:
                mask = np.all(point_cloud[:, :3] > self.min_bound, axis=1)
                point_cloud = point_cloud[mask]
            if self.max_bound is not None:
                mask = np.all(point_cloud[:, :3] < self.max_bound, axis=1)
                point_cloud = point_cloud[mask]

        point_cloud = point_cloud_sampling(point_cloud, self.num_points, 'fps')
        
        depth = depth[::-1]
        
        return point_cloud, depth
        

    def get_visual_obs(self):

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
        }
        return obs_dict
            
            
    def step(self, action: np.array):

        raw_state, reward, done, env_info = self.env.step(action)
        
        self.cur_step += 1
        
        ##change the camera pose
        # self.env.sim.model.cam_pos[2] = [random.uniform(-0.5, 0.5), random.uniform(-1.5, 1.5), random.uniform(-1, 1)]   
        # self.pc_generator.cam_pos = self.env.sim.model.cam_pos[2]
            ## 随机调整相机角度
        # 原始四元数
        def mjc2sci_quat(mjc_quat):
            """_summary_

            Args:
                mjc_quat (_type_): quaternion in mujoco format [w, x, y, z]

            Returns:
                _type_: _description_
            """
            return [mjc_quat[1], mjc_quat[2], mjc_quat[3], mjc_quat[0]]
        def sci2mjc_quat(sci_quat):
            return [sci_quat[3], sci_quat[0], sci_quat[1], sci_quat[2]]
        original_quat = self.origin_quat 
        # print("original quat: ", original_quat) 
        original_rotation = R.from_quat(mjc2sci_quat(original_quat))  # 将四元数转换为旋转对象

        # 随机生成欧拉角变动（限制在 ±30 度范围内）
        self.disturbance = 0
        disturbance = self.disturbance
        delta_roll = np.random.uniform(-disturbance, disturbance)  # 绕 x 轴旋转
        delta_pitch = np.random.uniform(-disturbance, disturbance)  # 绕 y 轴旋转
        delta_yaw = np.random.uniform(-disturbance, disturbance)  # 绕 z 轴旋转
        delta_rotation = R.from_euler('xyz', [delta_roll, delta_pitch, delta_yaw],degrees= True)  # 生成旋转变换
        new_rotation = original_rotation * delta_rotation  # 计算新的旋转
        # 计算新的相机位置
        # 打印新的旋转角度
        new_euler = new_rotation.as_euler('xyz', degrees=True)
        # print(f"New camera rotation (degrees): roll={new_euler[0]}, pitch={new_euler[1]}, yaw={new_euler[2]}")
        # 更新相机角度

        self.env.sim.model.cam_quat[2] = sci2mjc_quat(new_rotation.as_quat())
        
        self.pc_transform = R.from_matrix(self.old_pc_transform)*delta_rotation
        self.pc_transform = self.pc_transform.as_matrix()
        
        
        
        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:  # make channel first
            obs_pixels = obs_pixels.transpose(2, 0, 1)

        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_state,
        }
        
        #gsj
        self.agent_poses.append(robot_state)
        self.pcs.append(point_cloud)

        done = done or self.cur_step >= self.episode_length
        
        return obs_dict, reward, done, env_info

    def reset(self):
        self.env.reset()
        self.env.reset_model()
        raw_obs = self.env.reset()
        self.cur_step = 0

        obs_pixels = self.get_rgb()
        robot_state = self.get_robot_state()
        point_cloud, depth = self.get_point_cloud()
        
        if obs_pixels.shape[0] != 3:
            obs_pixels = obs_pixels.transpose(2, 0, 1)
        
        obs_dict = {
            'image': obs_pixels,
            'depth': depth,
            'agent_pos': robot_state,
            'point_cloud': point_cloud,
            'full_state': raw_obs,
        }


        return obs_dict

    def seed(self, seed=None):
        pass

    def set_seed(self, seed=None):
        pass

    def render(self, mode='rgb_array'):
        img = self.get_rgb()
        return img

    def close(self):
        pass

