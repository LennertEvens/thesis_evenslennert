from gymnasium.envs.registration import register

register(
    id='gd_env-v0',
    entry_point='gd_env.envs:GradDescentEnv',
)