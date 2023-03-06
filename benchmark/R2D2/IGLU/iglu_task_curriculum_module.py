from typing import Dict, List 
from collections import OrderedDict

import numpy as np
from regym.modules import Module
#from iglu.tasks import RandomTasks
from gridworld.tasks import RandomTasks

import wandb

import gym

class IGLUTaskCurriculumWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_nbr_max_blocks = None
        self.current_height_levels = None
        self.allow_float = None
        self.current_max_dist = None
        self.current_max_nbr_unique_colors = None
         
    def update_curriculum(
        self,
        max_blocks,
        height_levels,
        allow_float,
        max_dist, 
        num_colors,
        ):
        
        self.current_nbr_max_blocks=max_blocks
        self.current_height_levels=height_levels
        self.allow_float=allow_float
        self.current_max_dist=max_dist 
        self.current_max_nbr_unique_colors=num_colors
        
        self.reset_counters()

    def reset_counters(self):
        self.used_blocks_counter = 0
        
    def reset(self, **kwargs):
        self.reset_counters()
        _ = self.env.reset(**kwargs)
        
        self.lower_cam_action = OrderedDict({
            'attack': np.array(0),
            'back': np.array(0),
            'camera': np.array([15.0, 0]),
            'forward': np.array(0),
            'hotbar': np.array(0),
            'jump': np.array(0),
            'left': np.array(0),
            'right': np.array(0),
            'use': np.array(0),
        })

        obs, reward, done, info = self.env.step(self.lower_cam_action)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        if action['use'].item() == 1:
            if obs['grid'][self.current_height_levels:, ...].sum().item() > 0:
                done = True 

        self.used_blocks_counter = (obs['grid']>0).sum().item()
        
        """
        Previously: >=
        Now: the hypothesis is that by checking if the number of blocks
        used is greater than the limit, the agent can actually experience
        seeing any block and associate their colours to the action (rather
        than having to learn this later during the curriculum), and also
        it enables it to remove blocks and change the colour over one
        episode.
        """
        #if self.used_blocks_counter >= self.current_nbr_max_blocks:
        if self.used_blocks_counter > self.current_nbr_max_blocks:
            done = True

        return obs, reward, done, info
                     

def build_IGLUTaskCurriculumModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Dict[str,str]=None,
    ) -> Module:
    return IGLUTaskCurriculumModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids,
    )


class IGLUTaskCurriculumModule(Module):
    def __init__(
        self,
        id:str,
        config:Dict[str,object],
        input_stream_ids:Dict[str,str]=None,
        ):
        if input_stream_ids is None:
            input_stream_ids = {
                "logs_dict":"logs_dict",
                "RL_env":"modules:MARLEnvironmentModule_0:ref",
            }

        assert "max_episode_length" in config,\
        "IGLUTaskCurriculumModule relies on 'max_episode_length'.\n\
        Not found in config."
        assert "task" in config,\
        "IGLUTaskCurriculumModule relies on 'task'.\n\
        Not found in config."
        
        super(IGLUTaskCurriculumModule, self).__init__(
            id=id,
            type="IGLUTaskCurriculumModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )
        
        self.update_count = 0
        self.task = self.config['task']

        """
        self.episode_length_threshold = 0.6*self.config["max_episode_length"]
        self.run_mean_episode_length = self.config["max_episode_length"]
        """
        self.episode_success_threshold = 0.7
        self.run_mean_episode_success = 0.0
        self.run_mean_window = 50
        self.reset_running_mean()

        self.current_nbr_max_blocks = 1

        #TODO: find a better way to specify the curriculum:
        self.height_levels_block_period = 4
        self.current_height_levels = 1
        
        # TODO: check whether this allow_float
        # is impairing or not?
        # For now, it is set to false, because
        # the height_levels param does not 
        # constrain the blocks to be lower
        # than a maximal height, only to 
        # occupy less than height_levels levels.
        self.allow_float = False
        
        self.current_max_dist = 11
        self.current_max_nbr_unique_colors = 6
        
        # Initialisation:
        self.task.env.launch_env_processes()
        for env in self.task.env.env_processes:#+self.task.test_env.env_processes:
            env.set_task_generator(
                RandomTasks(
                    max_blocks=self.current_nbr_max_blocks,
                    height_levels=self.current_height_levels,
                    allow_float=self.allow_float,
                    max_dist=self.current_max_dist, 
                    num_colors=self.current_max_nbr_unique_colors,
                )
            )
            # Updating wrapper's vitals:
            while not isinstance(env, IGLUTaskCurriculumWrapper):
                env = env.env
            env.update_curriculum(
               max_blocks=self.current_nbr_max_blocks,
               height_levels=self.current_height_levels,
               allow_float=self.allow_float,
               max_dist=self.current_max_dist, 
               num_colors=self.current_max_nbr_unique_colors,
            ) 
    
    def reset_running_mean(self):
        self.prev_mean_episode_successes = [0.0]*self.run_mean_window

    def save(self, path):
        torch.save(self, os.path.join(path, self.id+".module"))

    def load(self, path):
        self = torch.load(os.path.join(path, self.id+".module"))

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        Operates on inputs_dict that is made up of referents to the available stream.
        Make sure that accesses to its element are non-destructive.

        :param input_streams_dict: dict of str and data elements that 
            follows `self.input_stream_ids`'s keywords and are extracted 
            from `self.input_stream_keys`-named streams.

        :returns:
            - outputs_stream_dict: 
        """
        outputs_stream_dict = {}

        logs_dict = input_streams_dict["logs_dict"]
        RL_env = input_streams_dict["RL_env"]
        RL_env_outputs = RL_env.outputs_stream_dict

        new_trajectory_batch_published = RL_env_outputs["new_trajectories_published"]

        if new_trajectory_batch_published:
            # Compute new running mean episode length:
            curr_mean_episode_success = RL_env_outputs["PerEpisodeBatch/MeanEpisodeSuccess"]
            self.prev_mean_episode_successes.append(curr_mean_episode_success)
            self.prev_mean_episode_successes.pop(0)

            self.run_mean_episode_success = np.mean(self.prev_mean_episode_successes)
            
            # Test whether RM ep. success is above threshold:
            if self.run_mean_episode_success >= self.episode_success_threshold: 
                print(f"IGLUTaskCurriculumModule: mean_episode_success:{self.run_mean_episode_success} >= threshold: {self.episode_success_threshold}")
                # Reset running mean:
                self.reset_running_mean()
                # and increase the number of blocks:
                self.current_nbr_max_blocks += 1

                #TODO: find a better way to specify the curriculum:
                if self.current_nbr_max_blocks % self.height_levels_block_period == 0:
                    self.current_height_levels += 1

                # Update envs:
                for env_idx, env in enumerate(self.task.env.env_processes):#+self.task.test_env.env_processes:
                    print(f"IGLUTaskCurriculumModule: UPDATING TASKSET for env {env_idx+1}/{len(self.task.env.env_processes)} : ....")
                    # Updating Taskset:
                    env.set_task_generator(
                        RandomTasks(
                            max_blocks=self.current_nbr_max_blocks,
                            height_levels=self.current_height_levels,
                            allow_float=self.allow_float,
                            max_dist=self.current_max_dist, 
                            num_colors=self.current_max_nbr_unique_colors,
                        )
                    )
                    # Updating wrapper's vitals:
                    while not isinstance(env, IGLUTaskCurriculumWrapper):
                        env = env.env
                    env.update_curriculum(
                       max_blocks=self.current_nbr_max_blocks,
                       height_levels=self.current_height_levels,
                       allow_float=self.allow_float,
                       max_dist=self.current_max_dist, 
                       num_colors=self.current_max_nbr_unique_colors,
                    ) 
                    print(f"IGLUTaskCurriculumModule: UPDATING TASKSET for env {env_idx+1}/{len(self.task.env.env_processes)} : DONE.")
            
            datad = {
                "IGLUTaskCurriculum/RunningMeanEpisodeSuccess": self.run_mean_episode_success,
                "IGLUTaskCurriculum/EpisodeSuccessThreshold": self.episode_success_threshold,
                "IGLUTaskCurriculum/RunningMeanWindowSize": self.run_mean_window,
                "IGLUTaskCurriculum/MaxNbrBlocks":self.current_nbr_max_blocks,
                "IGLUTaskCurriculum/HeightLevels":self.current_height_levels,
                "IGLUTaskCurriculum/MaxDist":self.current_max_dist,
                "IGLUTaskCurriculum/MaxNbrUniqueColors":self.current_max_nbr_unique_colors,
            }

            wandb.log(datad, commit=False)
        
        return outputs_stream_dict
 