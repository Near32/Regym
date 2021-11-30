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

        default_input_stream_ids = {
            "logs_dict":"logs_dict",
            "losses_dict":"losses_dict",
            "epoch":"signals:epoch",
            "mode":"signals:mode",

            "reset_actors":"modules:marl_environment_module:reset_actors",
            
            "observations":"modules:marl_environment_module:ref:player_0:observations",
            "infos":"modules:marl_environment_module:ref:player_0:infos",
            "actions":"modules:marl_environment_module:ref:player_0:actions",
            "succ_observations":"modules:marl_environment_module:ref:player_0:succ_observations",
            "succ_infos":"modules:marl_environment_module:ref:player_0:succ_infos",
            "rewards":"modules:marl_environment_module:ref:player_0:rewards",
            "dones":"modules:marl_environment_module:ref:player_0:dones",
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

        if hasattr(self, 'observations')\
        and self.agent.training:
            self.agent.handle_experience(
                s=self.observations,
                a=self.actions,
                r=input_streams_dict['rewards'],
                succ_s=self.new_observations,
                done=input_streams_dict['dones'],
                infos=self.infos,
                succ_infos=copy.deepcopy(self.new_infos),
            )

        # TODO: maybe reset everything if no attr observations:
        for actor_index in input_streams_dict['reset_actors']:
            self.agent.reset_actors(indices=[actor_index])                

        self.new_actions = self.agent.take_action(
            state=self.new_observations,
            infos=self.new_infos, 
        ) \
        if self.agent.training else \
        self.agent.query_action(
            state=self.new_observations,
            infos=self.new_infos,
        )

        self.observations = copy.deepcopy(self.new_observations)
        self.infos = copy.deepcopy(self.new_infos)
        self.actions = copy.deepcopy(self.new_actions)

        outputs_streams_dict[self.config['actions_stream_id']] = copy.deepcopy(self.new_actions)
        outputs_streams_dict["signals:agent_update_count"] = self.agent.get_update_count()

        return outputs_streams_dict
