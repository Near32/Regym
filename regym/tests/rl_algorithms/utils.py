from regym.rl_loops import singleagent_loops
import numpy as np

def can_act_in_environment(task, build_agent_function, config_dict, name, num_actions=5):
    env = task.env
    agent = build_agent_function(task, config_dict, name)
    observation = env.reset()
    for _ in range(num_actions):
        action = agent.take_action(observation)
        observation, _, _, _ = env.step(action)
        assert env.action_space.contains(action[0])

def learns_to_solve_task(nbr_episode, task, build_agent_function, config_dict, name):
    agent = build_agent_function(task, config_dict, name)
    assert agent.training, 'Agent should be training in order to solve test environment'
    import tqdm
    progress_bar = tqdm.tqdm(range(nbr_episode))
    running_mean_episode_length = 0
    running_window = 10
    for step in progress_bar:
        trajectory = task.run_episode(agent, training=True)
        running_mean_episode_length = (running_mean_episode_length*(running_window-1)+np.mean([len(t) for t in trajectory]))/(running_window)
        progress_bar.set_description(f'{agent.name} in {task.name}. Episode length: {running_mean_episode_length}')

