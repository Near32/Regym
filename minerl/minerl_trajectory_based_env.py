from typing import Generator, Callable

import numpy as np
import gym


def trajectory_based_rl_loop(agent, minerl_trajectory_env: gym.Env):
    '''
    Feeds :param: agent a sequence of experiences coming from
    :param: minerl_trajectory_env.

    The agent is requested to take actions, but these are ignored by
    the underlying environment. This is because as all observations,
    rewards and actions are stored in the trajectory
    underlying :param: minerl_trajectory_env.

    :param agent: Ideally R2D2 agent. Off-policy agent.
    :param minerl_trajectory_env: Environment that's going to be fed to agent
    '''
    original_nbr_actor = agent.nbr_actor
    agent.nbr_actor = 1  # During this fake trajectory, we only want 1 actor

    obs = minerl_trajectory_env.reset()
    done = False
    while not done:
        # With r2d2, we explicitly need to have 'extra_inputs' in frame (rnn) state
        agent.rnn_states['phi_body']['extra_inputs'] = {}

        # Taking an action is mandatory to propagate rnn_states, even if
        # action is ignored
        _ = agent.take_action(np.expand_dims(obs, 0))

        succ_obs, reward, done, info = env.step(action=None)
        agent.handle_experience(np.expand_dims(obs, 0),
                                np.expand_dims(info['a'], 0),
                                np.expand_dims(reward, 0),
                                np.expand_dims(succ_obs, 0),
                                np.expand_dims(done, 0),
                                infos=[info])
        obs = succ_obs
    # Upon trajectory termination, we reset number of actors
    agent.nbr_actor = original_nbr_actor


class MineRLTrajectoryBasedEnv(gym.Env):

    def __init__(self, trajectory_generator: Generator,
                 action_parser: Callable = lambda x: x):
        '''
        :param trajectory_generator: Generator obtained by running
            >>> data_pipeline = minerl.data.make(MineRLEnvName)
            >>> traj_names = data_pipeline.get_trajectory_names()
            >>> data_generator = data_pipeline.load_data(traj_names[any_will_do])

        :param action_parser: Function that processes action. Default is identity.
                              Useful to transform continuous actions into discretized ones
        '''

        self.trajectory_generator = trajectory_generator
        self.action_parser = action_parser

        # Format: (observation, action, reward, succ_observation, done)
        self.current_experience: Tuple = None

        # Generator cannot be resetted, so we need to ensure that reset is only
        # called once.
        self.has_reset_been_called = False
        self.is_trajectory_done = False

    def reset(self):
        if self.has_reset_been_called:
            raise NotImplementedError('Resetting MineRLTrajectoryBasedEnv more than ONCE is not supported')
        self.has_reset_been_called = True

        (o, a, r, succ_o, d) = next(self.trajectory_generator)
        self.current_experience = (o, a, r, succ_o, d)
        # NOTE: shape should be (channels, height, width)
        return o['pov']

    def step(self, action):
        '''
        We ignore :param: action, because we are following a fixed trajectory
        info should contain the action in self.current_experience after being parsed!
        '''
        if self.is_trajectory_done:
            return None

        # Decide what we return
        (o, a, r, succ_o, d) = self.current_experience

        info = {'a': self.action_parser(a['vector']),
 'inventory_vector': o['vector']
               }

        # Update current experience
        self.is_trajectory_done = d
        if not self.is_trajectory_done:
            self.current_experience = next(self.trajectory_generator)

        return succ_o['pov'], r, d, info

    def render(self):
        raise NotImplementedError('Not supported')


if __name__ == "__main__":
    import minerl
    # Preliminaries
    data_pipeline = minerl.data.make('MineRLTreechopVectorObf-v0')
    traj_names = data_pipeline.get_trajectory_names()
    data_iterator = data_pipeline.load_data(traj_names[0])

    import torch
    agent = torch.load('./test_agent.pt')

    # Making the env
    env = MineRLTrajectoryBasedEnv(data_iterator)
    trajectory_based_rl_loop(agent, env)
