from typing import Callable, Dict, Optional

import torch
import torch.nn.functional as F

from .base import ForwardPassResult


class ReinforceEstimator:
    def __init__(
        self,
        reward_scale: float = 1.0,
        use_baseline: bool = True,
        baseline_beta: float = 0.9,
    ):
        self.reward_scale = reward_scale
        self.use_baseline = use_baseline
        self.baseline_beta = baseline_beta
        self._baseline: Optional[torch.Tensor] = None

    def _update_baseline(self, loss_value: torch.Tensor) -> torch.Tensor:
        loss_detached = loss_value.detach()
        if self._baseline is None:
            self._baseline = loss_detached
        else:
            self._baseline = (1 - self.baseline_beta) * self._baseline + self.baseline_beta * loss_detached
        return self._baseline

    def build_objective(
        self,
        loss_value: torch.Tensor,
        log_prob_sum: torch.Tensor,
        update_baseline: bool = True,
    ) -> (torch.Tensor, Dict[str, Optional[torch.Tensor]]):
        baseline = None
        if self.use_baseline:
            if update_baseline:
                baseline = self._update_baseline(loss_value)
            else:
                baseline = self._baseline
            if baseline is not None:
                advantage = loss_value.detach() - baseline
            else:
                advantage = loss_value.detach()
        else:
            advantage = loss_value.detach()

        policy_loss = self.reward_scale * (advantage * log_prob_sum.mean())

        estimator_state: Dict[str, Optional[torch.Tensor]] = {
            "reinforce_advantage": advantage.detach(),
            "reinforce_baseline": baseline.detach() if baseline is not None else None,
            "reinforce_log_prob_mean": log_prob_sum.mean().detach(),
        }

        return policy_loss, estimator_state


def reinforce_forward_pass(
    *,
    model,
    tokenizer,
    loss_instance,
    learnable_inputs: torch.Tensor,
    embedding_weights_subset: torch.Tensor,
    allowed_tokens: torch.Tensor,
    target_length: int,
    batch_size: int,
    device: torch.device,
    model_precision: str,
    pre_prompt: Optional[str],
    filter_vocab: bool,
    reinforce_helper: ReinforceEstimator,
    update_baseline: bool = True,
) -> ForwardPassResult:
    message_logits = learnable_inputs.repeat(batch_size, 1, 1)
    if model_precision == "half":
        message_logits = message_logits.half()

    categorical_dist = torch.distributions.Categorical(logits=message_logits)
    message_ids = categorical_dist.sample()
    message_log_probs = categorical_dist.log_prob(message_ids)

    message_one_hot = F.one_hot(message_ids, num_classes=embedding_weights_subset.shape[0]).float()
    if model_precision == "half":
        message_one_hot = message_one_hot.half()

    input_embeddings = torch.matmul(message_one_hot, embedding_weights_subset)
    embedding_layer = model.get_input_embeddings()

    if pre_prompt is not None:
        pre_prompt_tokens = tokenizer(pre_prompt, return_tensors="pt").input_ids.to(device)
        pre_prompt_embeds = embedding_layer(pre_prompt_tokens).repeat(batch_size, 1, 1)
        current_embeds = torch.cat([pre_prompt_embeds, input_embeddings], dim=1)
    else:
        current_embeds = input_embeddings

    outputs = model(
        inputs_embeds=current_embeds,
        output_hidden_states=True,
        use_cache=True,
        return_dict=True,
    )

    logits = outputs.logits
    logits_allowed = logits[..., allowed_tokens]
    prompt_logits = logits_allowed
    past_key_values = outputs.past_key_values
    all_hidden_states = [outputs.hidden_states]

    all_logits = [logits_allowed[:, -1:, :]]

    completion_ids = []
    current_length = 1
    while current_length < target_length:
        next_token_id = torch.argmax(all_logits[-1], dim=-1)
        if filter_vocab:
            next_token_embedding = embedding_weights_subset[next_token_id]
        else:
            next_token_embedding = embedding_layer(next_token_id)

        completion_ids.append(next_token_id)

        outputs = model(
            inputs_embeds=next_token_embedding,
            past_key_values=past_key_values,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values

        next_logits = outputs.logits[..., allowed_tokens]
        all_logits.append(next_logits)
        all_hidden_states.append(outputs.hidden_states)

        current_length += 1

    generated_logits = torch.cat(all_logits, dim=1)
    completion_ids = torch.cat(completion_ids, dim=1)

    losses_dict = loss_instance.compute_loss(
        input_dict={
            "generated_logits": generated_logits,
            "generated_hidden_states": all_hidden_states,
            "completion_ids": completion_ids,
            "prompt_ids": message_ids,
            "prompt_logits": prompt_logits,
        },
    )
    loss = losses_dict["sumloss"]

    log_prob_sum = message_log_probs.sum(dim=-1)
    backward_loss, estimator_state = reinforce_helper.build_objective(
        loss,
        log_prob_sum,
        update_baseline=update_baseline,
    )

    estimator_state["reinforce_log_prob_sum_mean"] = log_prob_sum.mean().detach()

    return ForwardPassResult(
        loss=loss,
        backward_loss=backward_loss,
        losses_dict=losses_dict,
        generated_logits=generated_logits,
        prompt_ids=message_ids,
        prompt_logits=prompt_logits,
        completion_ids=completion_ids,
        estimator_state=estimator_state,
    )


def estimate_reinforce_gradient_variance(
    num_samples: int,
    baseline_grad: Optional[torch.Tensor],
    forward_pass_fn: Callable[..., ForwardPassResult],
    learnable_inputs: torch.Tensor,
) -> Dict[str, float]:
    if num_samples < 2 or baseline_grad is None:
        return {}

    grad_mean = baseline_grad.clone()
    grad_m2 = torch.zeros_like(grad_mean)
    sample_count = 1

    for _ in range(num_samples - 1):
        sample_result = forward_pass_fn(update_baseline=False)
        sample_grad = torch.autograd.grad(
            sample_result.backward_loss,
            learnable_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].detach()

        sample_count += 1
        delta = sample_grad - grad_mean
        grad_mean = grad_mean + delta / sample_count
        grad_m2 = grad_m2 + delta * (sample_grad - grad_mean)

        del sample_result, sample_grad

    if sample_count < 2:
        return {}

    variance = grad_m2 / (sample_count - 1)
    return {
        "reinforce_grad_variance_mean": variance.mean().item(),
        "reinforce_grad_variance_max": variance.max().item(),
        "reinforce_grad_variance_norm": variance.norm().item(),
        "reinforce_grad_mean_norm": grad_mean.norm().item(),
        "reinforce_grad_variance_samples": int(sample_count),
    }
