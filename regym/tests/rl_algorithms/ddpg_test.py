from test_fixtures import ddpg_config_dict, PendulumTask
from utils import can_act_in_environment

from regym.rl_algorithms.agents import build_DDPG_Agent
from regym.rl_loops.singleagent_loops.rl_loop import run_episode


def test_reinforce_can_take_actions_continuous_obvservation_discrete_action(PendulumTask, ddpg_config_dict):
    can_act_in_environment(PendulumTask, build_DDPG_Agent, ddpg_config_dict, name=__name__)


def test_learns_to_solve_cartpole(PendulumTask, ddpg_config_dict):
    agent = build_DDPG_Agent(PendulumTask, ddpg_config_dict, 'DDPG-Reinforce')
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(4000))
    for _ in progress_bar:
        trajectory = run_episode(PendulumTask.env, agent, training=True)
        total_trajectory_reward = sum(map(lambda experience: experience[2], trajectory))
        progress_bar.set_description(f'{agent.name} in {PendulumTask.env.spec.id}. Episode length: {total_trajectory_reward}')
