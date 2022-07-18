import gym

class EnvironmentCreator():
    def __init__(self, environment_name_cli, is_unity_environment, is_gym_environment, wrapping_fn=None, env_config={}):
        self.environment_name = environment_name_cli
        self.is_unity_environment = is_unity_environment
        self.is_gym_environment = is_gym_environment
        self.wrapping_fn = wrapping_fn
        self.env_config = env_config

    def __call__(self, worker_id=None, seed=0):
        if self.is_gym_environment:
            env = gym.make(self.environment_name, **self.env_config)
            env.seed(seed)
            if self.wrapping_fn is not None: env = self.wrapping_fn(env=env)
            return env
        if self.is_unity_environment: 
            if 'obstacletower' not in self.environment_name: raise ValueError('Only obstacletower environment currently supported')
            from obstacle_tower_env import ObstacleTowerEnv
            if worker_id is None: worker_id=0
            return ObstacleTowerEnv(self.environment_name, retro=True, realtime_mode=False, timeout_wait=60, worker_id=worker_id) #timeout_wait=6000,  # retro=True mode creates an observation space of a 64x64 (Box) image
