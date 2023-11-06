from gym.envs.registration import register

register(
    id='gd_env_eval-v0',
    entry_point='gd_env_eval.envs:GradDescentEnv_eval',
)