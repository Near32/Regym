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

register(
	id='TinyAbstractHanabi2P2C3A-v0',
	entry_point='regym.environments.envs.hanabi.envs:TinyAbstractHanabi2P2C3AGymEnv',
)


register(
	id='TinyAbstractHanabi2P2C3A-OHEObs-v0',
	entry_point='regym.environments.envs.hanabi.envs:TinyAbstractHanabiOHEObs2P2C3AGymEnv',
)

register(
    id='EasyTinyAbstractHanabi2P2C3A-OHEObs-v0',
    entry_point='regym.environments.envs.hanabi.envs:EasyTinyAbstractHanabiOHEObs2P2C3AGymEnv',
)