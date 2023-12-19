import os
import gymnasium as gym
from stable_baselines3 import TD3, SAC, DDPG, HerReplayBuffer, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure, read_csv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise, NormalActionNoise
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Callable
import gd_env
import torch
import subprocess
import numpy as np
# CUDA
print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

# Storing ID of current CUDA device
cuda_id = torch.cuda.current_device()
print(f"ID of current CUDA device:{torch.cuda.current_device()}")

print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")

env_id = "gd_env-v0"
eval_env_id = "gd_env-v0"
n_training_envs = 1
n_eval_envs = 1

# Create log dir where evaluation results will be saved
eval_log_dir = "eval_logs/"
os.makedirs(eval_log_dir, exist_ok=True)

# Initialize a vectorized training environment with default parameters
train_env = gym.make(env_id)
train_env = DummyVecEnv([lambda: train_env])
eval_env = gym.make(env_id,mode='eval')
eval_env = DummyVecEnv([lambda: eval_env])

# train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0, mode='train')

# Separate evaluation env, with different parameters passed via env_kwargs
# Eval environments can be vectorized to speed up evaluation.
# eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0, mode='eval')

# Create callback that evaluates agent for 5 episodes every 500 training environment steps.
# When using multiple training environments, agent will be evaluated every
# eval_freq calls to train_env.step(), thus it will be evaluated every
# (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                              log_path=eval_log_dir, eval_freq=max(10000 // n_training_envs, 1),
                              n_eval_episodes=1, deterministic=True,
                              render=False)

# set up logger
tmp_path = "/tmp/sb3_log/"
os.system('rm -rf /tmp')
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
action_noise = NormalActionNoise(np.array([0.0]),np.array([0.15]))

# action_noise = OrnsteinUhlenbeckActionNoise(np.array([0.0]),np.array([0.01]))
model = TD3("MlpPolicy", train_env,batch_size=2000,verbose=1,learning_rate=1e-4,gamma=0.99,seed=0,tau=0.005,
            train_freq=50,policy_delay=2,policy_kwargs=dict(net_arch=[128,128,128]),action_noise=action_noise,device=cuda_id, buffer_size=int(5e6))

model.set_logger(new_logger)
model.learn(total_timesteps=3e6, callback=eval_callback,progress_bar=True)

model.save("gd2")
