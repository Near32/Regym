import numpy as np
import gym
import cv2 

from collections import deque

# # Wrappers:
# # Observation Wrappers:
'''
Adapted from:
https://github.com/chainer/chainerrl/blob/master/chainerrl/wrappers/atari_wrappers.py
'''
class LazyFrames(object):
    def __init__(self, frames):
        self._frames = frames
    
    def __array__(self, dtype=None):
        out = np.concatenate(self._frames, axis=-1)
        if dtype is not None:
            out = out.astype(dtype)
        return out


class RandNoOpStartWrapper(gym.Wrapper):
    def __init__(self, env, nbr_max_random_steps=30):
        gym.Wrapper.__init__(self,env)
        self.nbr_max_random_steps = nbr_max_random_steps
        self.total_reward = 0
        
    def reset(self, **args):
        obs = self.env.reset()
        nbr_rand_noop = random.randint(0, self.nbr_max_random_steps)
        self.total_reward = 0
        for _ in range(nbr_rand_noop):
            # Execute No-Op:
            obs, r, d, i = self.env.step(0)
            self.total_reward += r
        return obs 

    def step(self, action):
        obs, r, d, info = self.env.step(action=action)
        if self.total_reward != 0:
            r += self.total_reward
            self.total_reward = 0
        return obs, r, d, info 


class SingleLifeWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.done = False
        self.life_done = True 

        AtariEnv = env
        while True:
            env = getattr(AtariEnv, 'env', None)
            if env is not None:
                AtariEnv = env
            else:
                break
        self.AtariEnv = AtariEnv
        self.lives = self.AtariEnv.ale.lives()

    def reset(self, **args):
        self.done = False
        self.lives = self.env.env.ale.lives()
        return self.env.reset(**args)

    def step(self, action):
        if self.done:
            self.reset()
        obs, reward, done, info = self.env.step(action)

        force_done = done
        if self.life_done:
            if self.lives > info['ale.lives']:
                force_done = True
                self.lives = info['ale.lives']
        
        if force_done:
            reward = -1
    
        self.done = done
        info['real_done'] = done
        
        return obs, reward, force_done, info 
        

class FrameSkipStack(gym.Wrapper):
    """
    Return a stack of framed composed of every 'skip'-th repeat.
    The rewards are summed over the skipped and stackedd frames.
    
    This wrapper assumes:
    - the observation space of the environment to be frames solely.
    - the frames are concatenated on the last axis, i.e. the channel axis.
    """
    def __init__(self, env, skip=4, act_rand_repeat=False, stack=4, single_life_episode=False):
        gym.Wrapper.__init__(self,env)
        self.skip = skip if skip is not None else 1
        self.stack = stack if stack is not None else 1
        self.act_rand_repeat = act_rand_repeat
        self.single_life_episode = single_life_episode

        self.observations = deque([], maxlen=self.stack)
        
        assert(isinstance(self.env.observation_space, gym.spaces.Box))
        
        low_obs_space = np.repeat(self.env.observation_space.low, self.stack, axis=-1)
        high_obs_space = np.repeat(self.env.observation_space.high, self.stack, axis=-1)
        self.observation_space = gym.spaces.Box(low=low_obs_space, high=high_obs_space, dtype=self.env.observation_space.dtype)

        self.done = False

        if self.single_life_episode: 
            self.life_done = True 

            AtariEnv = env
            while True:
                env = getattr(AtariEnv, 'env', None)
                if env is not None:
                    AtariEnv = env
                else:
                    break
            self.AtariEnv = AtariEnv
            self.lives = self.AtariEnv.ale.lives()
        
    def _get_obs(self):
        assert(len(self.observations) == self.stack)
        return LazyFrames(list(self.observations))
        
    def reset(self, **args):
        obs = self.env.reset()
        
        self.done = False
        
        if self.single_life_episode:
            self.lives = self.AtariEnv.ale.lives()
        
        for _ in range(self.stack):
            self.observations.append(obs)
        return self._get_obs()
    
    def step(self, action):
        if self.done:
            self.reset()
        
        total_reward = 0.0
        nbr_it = self.skip
        if self.act_rand_repeat:
            nbr_it = random.randint(1, nbr_it)

        for i in range(nbr_it):
            obs, reward, done, info = self.env.step(action)

            force_done = done
            if self.single_life_episode:
                if self.life_done:
                    if self.lives > info['ale.lives'] and info['ale.lives'] > 0:
                        force_done = True
                        self.lives = info['ale.lives']
                
                if reward < 0:
                    force_done = True
                elif force_done:
                    reward = -1
            
                info['real_done'] = done

            total_reward += reward

            if self.act_rand_repeat:
                self.observations.append(obs)

            self.done = done
            if done or force_done:
                break
            
        if not(self.act_rand_repeat):
            self.observations.append(obs)
        
        return self._get_obs(), total_reward, force_done, info

'''
class GrayScaleObservation(gym.ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale. """
    def __init__(self, env, keep_dim=True):
        super(GrayScaleObservation, self).__init__(env)
        self.keep_dim = keep_dim

        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3
        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            observation = np.expand_dims(observation, -1)
        return observation
'''


def atari_pixelwrap(env, size, skip=None, act_rand_repeat=False, stack=None, grayscale=False, nbr_max_random_steps=0, single_life_episode=True):
    # Observations:
    if grayscale:
        env = GrayScaleObservation(env=env) 
    if nbr_max_random_steps > 0:
        env = RandNoOpStartWrapper(env=env, nbr_max_random_steps=nbr_max_random_steps)
    #env = PixelObservationWrapper(env=env)
    env = FrameResizeWrapper(env, size=size) 
    if skip is not None or stack is not None:
        env = FrameSkipStack(env=env, skip=skip, act_rand_repeat=act_rand_repeat, stack=stack, single_life_episode=single_life_episode)
    #if single_life:
    #    env = SingleLifeWrapper(env=env)
    return env


class GrayScaleObservation(gym.ObservationWrapper):
    r"""Convert the image observation from RGB to gray scale. """
    def __init__(self, env, keep_dim=True):
        _env = env
        if isinstance(env, gym.wrappers.time_limit.TimeLimit):
            _env = env.env
        _env._get_image = _env.ale.getScreenGrayscale
        _env._get_obs = _env.ale.getScreenGrayscale

        super(GrayScaleObservation, self).__init__(env)
        
        assert len(env.observation_space.shape) == 3 and env.observation_space.shape[-1] == 3
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(obs_shape[0], obs_shape[1], 1), dtype=np.uint8)
        
    def observation(self, observation):
        return observation


class FrameResizeWrapper(gym.ObservationWrapper):
    """
    """
    def __init__(self, env, size=(64, 64)):
        gym.ObservationWrapper.__init__(self, env=env)
        
        self.size = size
        if isinstance(self.size, int):
            self.size = (self.size, self.size)

        low = np.zeros((*self.size, self.env.observation_space.shape[-1]))
        high  = 255*np.ones((*self.size, self.env.observation_space.shape[-1]))
        
        self.observation_space = gym.spaces.Box(low=low, high=high)
    
    def observation(self, observation):
        obs = cv2.resize(observation, self.size)
        obs = obs.reshape(self.observation_space.shape)
        return obs


# https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L275
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class FrameStack(gym.Wrapper):
    def __init__(self, env, stack=4,):
        gym.Wrapper.__init__(self,env)
        self.stack = stack if stack is not None else 1
        self.observations = deque([], maxlen=self.stack)
        
        assert(isinstance(self.env.observation_space, gym.spaces.Box))
        
        low_obs_space = np.repeat(self.env.observation_space.low, self.stack, axis=-1)
        high_obs_space = np.repeat(self.env.observation_space.high, self.stack, axis=-1)
        self.observation_space = gym.spaces.Box(low=low_obs_space, high=high_obs_space, dtype=self.env.observation_space.dtype)

    def _get_obs(self):
        assert(len(self.observations) == self.stack)
        return LazyFrames(list(self.observations))
        
    def reset(self, **args):
        obs = self.env.reset()
        for _ in range(self.stack):
            self.observations.append(obs)
        return self._get_obs()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.observations.append(obs)        
        return self._get_obs(), reward, done, info


# https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L12
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1) #pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)


# https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L97
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# https://github.com/openai/baselines/blob/9ee399f5b20cd70ac0a871927a6cf043b478193f/baselines/common/atari_wrappers.py#L125
class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)



def baseline_atari_pixelwrap(env, size, skip=4, stack=4, grayscale=True,  single_life_episode=True, nbr_max_random_steps=30, clip_reward=True):
    if grayscale:
        env = GrayScaleObservation(env=env) 
    
    if nbr_max_random_steps > 0:
        env = NoopResetEnv(env, noop_max=nbr_max_random_steps)
    env = MaxAndSkipEnv(env, skip=skip)
    
    env = FrameResizeWrapper(env, size=size) 
    
    if single_life_episode:
        env = EpisodicLifeEnv(env)
    env = FrameStack(env, stack=stack)
    
    if clip_reward:
        env = ClipRewardEnv(env)

    return env
