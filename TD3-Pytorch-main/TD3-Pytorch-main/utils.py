import numpy as np


def evaluate_policy(env, model, render, net_width, turns=1):
    scores = 0
    for j in range(turns):
        s, done, ep_r, steps = env.reset(), False, 0, 0
        function_nb = env.get_function_nb()
        traj = np.array(s[0:2])
        while not done:
            # Take deterministic actions at test time
            a = model.select_action(s, net_width)
            s_prime, r, done, info = env.step(a)
            traj = np.append(traj,s_prime[0:2],axis=0)


            ep_r += r
            steps += 1
            s = s_prime
            if render: env.render()

        scores += ep_r
    return scores / turns, traj, function_nb


#Just ignore this function~
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#reward engineering for better training
def Reward_adapter(r, EnvIdex):
    # For Pendulum-v0
    if EnvIdex == 0:
        r = (r + 8) / 8

    # For LunarLander
    elif EnvIdex == 1:
        if r <= -100: r = -10

    # For BipedalWalker
    elif EnvIdex == 4 or EnvIdex == 5:
        if r <= -100: r = -1
    return r