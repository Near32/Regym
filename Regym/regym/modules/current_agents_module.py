from typing import Dict, List 

from regym.modules.module import Module


class CurrentAgentsModule(Module):
    def __init__(self, 
                 id="current_agents",
                 agents:List[object]=None):
        """
        This is a placeholder for the agents. It must not be part of any pipeline.
        
        :param id: str defining the ID of the module.
        :param agents: List of Agents.
        """

        super(CurrentAgentsModule, self).__init__(
            id=id, 
            type="CurrentAgentsModule",
            config=None,
            input_stream_ids=None
        )

        self.agents = agents 

    def set_agents(self, agents):
        self.agents = agents

    def parameters(self):
        params = []
        for agent in self.agents:
            params += agent.parameters()
        return params 
        
    def get_input_stream_ids(self):
        raise NotImplementedError 

    def compute(self, input_streams_dict:Dict[str,object]) -> Dict[str,object] :
        raise NotImplementedError