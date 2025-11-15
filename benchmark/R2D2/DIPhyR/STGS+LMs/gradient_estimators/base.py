from typing import Dict, NamedTuple, Optional

import torch


class ForwardPassResult(NamedTuple):
    loss: torch.Tensor
    backward_loss: torch.Tensor
    losses_dict: Dict[str, torch.Tensor]
    generated_logits: torch.Tensor
    prompt_ids: torch.Tensor
    prompt_logits: torch.Tensor
    completion_ids: torch.Tensor
    estimator_state: Dict[str, Optional[torch.Tensor]]
