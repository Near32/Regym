from typing import Dict, List, Optional 

import copy
import numpy as np

from regym.modules.module import Module

def build_BabyAIBotModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Optional[Dict[str,str]]=None) -> Module:
    return BabyAIBotModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class BabyAIBotModule(Module):
    def __init__(
        self, 
        id:str,
        config=Dict[str,object],
        input_stream_ids:Optional[Dict[str,str]]=None
    ):
        """
        This is a placeholder for the BabyAI Bot.
        """
        
        player_idx = config.get('player_idx', 0)
        default_input_stream_ids = {
            "logs_dict":"logs_dict",
            "losses_dict":"losses_dict",
            "epoch":"signals:epoch",
            "mode":"signals:mode",

            "reset_actors":"modules:marl_environment_module:reset_actors",
            
            "observations":f"modules:marl_environment_module:ref:player_{player_idx}:observations",
            "infos":f"modules:marl_environment_module:ref:player_{player_idx}:infos",
            "actions":f"modules:marl_environment_module:ref:player_{player_idx}:actions",
            "succ_observations":f"modules:marl_environment_module:ref:player_{player_idx}:succ_observations",
            "succ_infos":f"modules:marl_environment_module:ref:player_{player_idx}:succ_infos",
            "rewards":f"modules:marl_environment_module:ref:player_{player_idx}:rewards",
            "dones":f"modules:marl_environment_module:ref:player_{player_idx}:dones",
        }

        if input_stream_ids is None:
            input_stream_ids = default_input_stream_ids
        else:
            for default_id, default_stream in default_input_stream_ids.items():
                if default_id not in input_stream_ids:
                    input_stream_ids[default_id] = default_stream

        super(BabyAIBotModule, self).__init__(
            id=id, 
            type="BabyAIBotModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )

        self.agent = self.config["agent"]
        
        self.pinput_streams_dict = None
        self.missions = {}
        self.agents = {}
        self.last_actions = {}
        self.agents_initialized = {0:False}
        self.agents_reset_history = {0:True}

    def init_agents(self, missions, indices=None):
        self.nbr_agents = len(missions)
        for aidx, mission in missions.items():
            if indices is not None \
            and aidx not in indices:
                continue
            
            if aidx in self.agents_initialized \
            and self.agents_initialized[aidx]:   continue
            self.agents[aidx] = self.agent(mission)
            if self.agents_reset_history.get(aidx, True):
                self.agents_reset_history[aidx] = False
                self.last_actions[aidx] = None
            self.agents_initialized[aidx] = True
    
    def extract_missions_from_infos(self):
        for aidx, infos in enumerate(self.new_infos):
            self.missions[aidx] = infos['babyai_mission']

    def update_missions(self):
        for aidx, mission in self.missions.items():
            self.agents[aidx].mission = mission

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_streams_dict = {}
        
        self.new_observations = input_streams_dict['succ_observations']
        self.new_infos = input_streams_dict['succ_infos']
        
        '''
        # Re-init all the time:
        for aidx in self.agents_initialized:
            self.agents_initialized[aidx] = False                
        '''

        self.extract_missions_from_infos()
        for midx, missions in self.missions.items():
            if midx in self.agents_initialized: continue
            self.agents_initialized[midx] = False
        if not all(self.agents_initialized.values()):
            indices = [idx for idx,value in self.agents_initialized.items() if value==False]
            self.init_agents(self.missions, indices=indices)
        self.update_missions()

        self.new_actions = []
        for aidx in range(self.nbr_agents):
            if aidx in input_streams_dict['reset_actors']:
                # Nothing to do, just wait for the next iteration
                # to update the agent ...
                new_action = 0
            elif self.pinput_streams_dict is not None\
            and self.pinput_streams_dict['dones'] is not None\
            and self.pinput_streams_dict['dones'][aidx]:
                new_action = self.last_actions[aidx]
                if new_action is None:
                    new_action = 0
                elif not isinstance(new_action, int):
                    new_action = new_action[0]
            else:
                agent = self.agents[aidx]
                last_action = self.last_actions[aidx] 
                try:
                    n_action = agent.replan(last_action)
                    new_action = n_action.value
                except Exception as e:
                    self.agents_initialized[aidx] = False
                    self.agents_reset_history[aidx] = True
                    self.init_agents(self.missions, indices=[aidx])
                    last_action = self.last_actions[aidx]
                    agent = self.agents[aidx]
                    new_action = agent.replan(last_action).value
            self.new_actions.append(new_action)
        
        self.last_actions = copy.deepcopy(self.new_actions)
        self.new_actions = np.asarray(self.new_actions)
        self.new_actions = np.reshape(self.new_actions, (self.nbr_agents, 1))

        self.observations = copy.deepcopy(self.new_observations)
        self.infos = copy.deepcopy(self.new_infos)
        self.actions = copy.deepcopy(self.new_actions)

        # Prepare reset at the next iteration, when the infos will have been updated...
        if len(input_streams_dict['reset_actors']):
            assert all(
                [   input_streams_dict['dones'][aidx] 
                    for aidx in input_streams_dict['reset_actors']
                ]
            )
            for aidx in input_streams_dict['reset_actors']:
                self.agents_initialized[aidx] = False   
                self.agents_reset_history[aidx] = True             
        
        outputs_streams_dict[self.config['actions_stream_id']] = copy.deepcopy(self.new_actions)
        
        self.ppinput_streams_dict = copy.deepcopy(self.pinput_streams_dict)
        self.pinput_streams_dict = copy.deepcopy(input_streams_dict)

        return outputs_streams_dict
