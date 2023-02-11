from typing import Dict, List, Optional 

import copy

from regym.modules.module import Module

def build_RLAgentModule(
    id:str,
    config:Dict[str,object],
    input_stream_ids:Optional[Dict[str,str]]=None) -> Module:
    return RLAgentModule(
        id=id,
        config=config, 
        input_stream_ids=input_stream_ids
    )


class RLAgentModule(Module):
    def __init__(
        self, 
        id:str,
        config=Dict[str,object],
        input_stream_ids:Optional[Dict[str,str]]=None
        ):
        """
        This is a placeholder for an RL agent.
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

        super(RLAgentModule, self).__init__(
            id=id, 
            type="RLAgentModule",
            config=config,
            input_stream_ids=input_stream_ids,
        )

        self.agent = self.config["agent"]


    def set_agent(self, agent):
        self.agent = agent

    def parameters(self):
        return self.agent.parameters() 
        
    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        """
        """
        outputs_streams_dict = {}
        
        self.new_observations = input_streams_dict['succ_observations']
        self.new_infos = input_streams_dict['succ_infos']
        
        # Allow Imitation Learning if the action has been 
        # overriden by some other module :
        provided_actions = input_streams_dict['actions']
        if provided_actions is not None:
            self.actions = provided_actions

        if hasattr(self, 'observations') and self.agent.training:
            self.agent.handle_experience(
                s=copy.deepcopy(self.observations),
                a=copy.deepcopy(self.actions),
                r=copy.deepcopy(input_streams_dict['rewards']),
                succ_s=copy.deepcopy(self.new_observations),
                done=copy.deepcopy(input_streams_dict['dones']),
                infos=copy.deepcopy(self.infos),
                succ_infos=copy.deepcopy(self.new_infos),
            )

        if self.agent.training:
            self.new_actions = self.agent.take_action(
                state=self.new_observations,
                infos=self.new_infos, 
            )
        else:
            self.new_actions = self.agent.query_action(
                state=self.new_observations,
                infos=self.new_infos,
            )

        self.observations = copy.deepcopy(self.new_observations)
        self.infos = copy.deepcopy(self.new_infos)
        self.actions = copy.deepcopy(self.new_actions)

        outputs_streams_dict[self.config['actions_stream_id']] = copy.deepcopy(self.new_actions)
        outputs_streams_dict["signals:agent_update_count"] = self.agent.get_update_count()

        if len(input_streams_dict['reset_actors'])!=0:
            assert all([input_streams_dict['dones'][aidx] for aidx in input_streams_dict['reset_actors']])
            self.agent.reset_actors(indices=input_streams_dict['reset_actors'])                
         
        return outputs_streams_dict
