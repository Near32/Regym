from gym.envs.registration import register

register(
    id='Hanabi-Full-v0',
    entry_point='regym.environments.envs.hanabi.envs:HanabiFullGymEnv',
)

register(
    id='Hanabi-Small-v0',
    entry_point='regym.environments.envs.hanabi.envs:HanabiSmallGymEnv',
)

register(
    id='Hanabi-VerySmall-v0',
    entry_point='regym.environments.envs.hanabi.envs:HanabiVerySmallGymEnv',
)