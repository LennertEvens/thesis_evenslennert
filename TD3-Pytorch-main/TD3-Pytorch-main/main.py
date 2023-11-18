import numpy as np
import torch
import gymnasium as gym
from TD3 import TD3_Agent, ReplayBuffer, device
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
from utils import str2bool,Reward_adapter,evaluate_policy
import gd_env
import gd_env_eval
from visualizer import visualize
import matplotlib.pyplot as plt
from numpy import linalg as LA
import torch.nn.utils.prune as prune

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--EnvIdex', type=int, default=0, help='PV0, Lch_Cv2, Humanv2, HCv2, BWv3, BWHv3, gd_env-v0')
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=30000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')
parser.add_argument('--Max_train_steps', type=int, default=5e6, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=5e3, help='Model saving interval, in steps.') #1e5
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--policy_delay_freq', type=int, default=1, help='Delay frequency of Policy Updating')
parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=64, help='Hidden net width') #256
parser.add_argument('--a_lr', type=float, default=1e-6, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=1e-6, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=16, help='batch_size of training') #256
parser.add_argument('--exp_noise', type=float, default=0.15, help='explore noise')#0.15
parser.add_argument('--noise_decay', type=float, default=0.999, help='Decay rate of explore noise')#0.998
opt = parser.parse_args()
print(opt)



def main():
    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
 
    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device:{torch.cuda.current_device()}")

    print(f"Name of current CUDA device:{torch.cuda.get_device_name(cuda_id)}")
    
    EnvName = ['Pendulum-v1','LunarLanderContinuous-v2','Humanoid-v2','HalfCheetah-v2','BipedalWalker-v3','BipedalWalkerHardcore-v3', 'gd_env-v0']
    BrifEnvName = ['PV1', 'LLdV2', 'Humanv2', 'HCv2','BWv3', 'BWHv3', 'gd_env-v0'] #Brief Environment Name.
    env_with_dw = [False, True, True, False, True, True, False]  # dw:die and win
    EnvIdex = opt.EnvIdex
    env = gym.make(EnvName[EnvIdex])
    # eval_env = gym.make(EnvName[EnvIdex])
    eval_env = gym.make('gd_env_eval-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    expl_noise = opt.exp_noise
    # max_e_steps = env._max_episode_steps
    max_e_steps = 10
    print('Env:', EnvName[EnvIdex], '  state_dim:', state_dim, '  action_dim:', action_dim, '  max_a:', max_action,
          '  min_a:', env.action_space.low[0],'  max_e_steps:',max_e_steps )

    update_after = 2 * max_e_steps  # update actor and critic after update_after steps
    start_steps = 10*max_e_steps #start using actor to iterate after start_steps steps

    #Random seed config:
    random_seed = opt.seed
    print("Random Seed: {}".format(random_seed))
    torch.manual_seed(random_seed)
    # env.seed(random_seed)
    # eval_env.seed(random_seed)
    np.random.seed(random_seed)
    

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(BrifEnvName[EnvIdex]) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    kwargs = {
        "env_with_dw":env_with_dw[EnvIdex],
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "gamma": opt.gamma,
        "net_width": opt.net_width,
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size":opt.batch_size,
        "policy_delay_freq":opt.policy_delay_freq
    }
    if not os.path.exists('model'): os.mkdir('model')
    model = TD3_Agent(**kwargs)
    if opt.Loadmodel: model.load(BrifEnvName[EnvIdex],opt.ModelIdex)
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=int(1e6))

    if opt.render:

        score, traj, _ = evaluate_policy(env, model, opt.render, turns=1)
        print('EnvName:', BrifEnvName[EnvIdex], 'score:', score)
        traj = np.reshape(traj,(int(np.size(traj,0)/2),2))
        np.savetxt("TD3_trajectory.txt", traj, fmt='%4.15f', delimiter=' ') 
        data = np.append(np.linspace(1,np.size(traj,0),np.size(traj,0)),LA.norm(traj,axis=1),axis = 0)
        data = np.reshape(data,(int(np.size(data)/2),2),order='F')
        np.savetxt("TD3_data.txt", data, fmt='%4.15f', delimiter=' ') 
        # plt.semilogy(np.linspace(1,np.size(traj,0),np.size(traj,0)),LA.norm(traj,axis=1))
        # plt.grid()
        # plt.show()
    else:
        total_steps = 0
        while total_steps < opt.Max_train_steps:
            s, done, steps, ep_r = env.reset(), False, 0, 0
            '''Interact & train'''
            while not done:
                steps += 1  #steps in one episode

                if total_steps < start_steps:
                    a = env.action_space.sample()
                else:
                    file = open("max_step.txt", "r")
                    line = file.readlines()
                    max_action = float(np.fromstring(line[0], dtype=float, sep=' '))
                    temp = model.select_action(s)
                    a = (temp + np.random.normal(0, max_action * expl_noise, size=action_dim)
                         ).clip(-max_action, max_action)  # explore: deterministic actions + noise
                s_prime, r, done, _, _ = env.step(a)
                r = Reward_adapter(r, EnvIdex)

                '''Avoid impacts caused by reaching max episode steps'''
                if (done and steps != max_e_steps):
                    dw = True  # dw: dead and win
                else:
                    dw = False

                replay_buffer.add(s, a, r, s_prime, dw)
                s = s_prime
                ep_r += r

                '''train if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= update_after and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every):
                        model.train(replay_buffer)
                
                '''record & log'''
                if total_steps % opt.eval_interval == 0:
                    expl_noise *= opt.noise_decay
                    score, _, _ = evaluate_policy(eval_env, model, False)
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('expl_noise', expl_noise, global_step=total_steps)
                    print('EnvName:', BrifEnvName[EnvIdex], 'steps: {}k'.format(int(total_steps/1000)), 'score:', score)
                total_steps += 1

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    model.save(BrifEnvName[EnvIdex],total_steps)
        env.close()


if __name__ == '__main__':
    main()




