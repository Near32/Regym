from typing import Callable, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ForwardPassResult


class STGS(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        stgs_hard: bool = False,
        init_temperature: float = 1.0,
        learnable_temperature: bool = False,
        nbr_learnable_temperatures: Optional[int] = None,
        conditioning_dim: int = 0,
        eps: float = 1e-12,
        device: str = "cpu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.stgs_hard = stgs_hard
        self.init_temperature = init_temperature
        self.learnable_temperature = learnable_temperature
        self.nbr_learnable_temperatures = nbr_learnable_temperatures
        if self.learnable_temperature:
            if self.nbr_learnable_temperatures is None:
                self.nbr_learnable_temperatures = 1
        self.conditioning_dim = conditioning_dim
        self.eps = eps
        self.device = device

        if self.learnable_temperature:
            if self.conditioning_dim < 1:
                self.temperature_param = nn.Parameter(torch.rand(self.nbr_learnable_temperatures, requires_grad=True, device=self.device))
            else:
                self.tau_fc = nn.Sequential(
                    nn.Linear(self.conditioning_dim, self.nbr_learnable_temperatures, bias=False),
                    nn.Softplus(),
                ).to(device=device)

    def forward(self, x: torch.Tensor, hidden_states: Optional[torch.Tensor] = None):
        if self.learnable_temperature:
            if self.conditioning_dim < 1:
                eff_temperature = self.eps + 1.0 / (F.softplus(self.temperature_param) + 1.0 / (self.eps + self.init_temperature))
                eff_temperature = eff_temperature.reshape(1, -1, 1)
                batch_size = x.shape[0]
                eff_temperature = eff_temperature.repeat(batch_size, 1, 1)
            else:
                assert hidden_states is not None
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                last_hidden_state = hidden_states[-1][:, -1, :].reshape(batch_size, self.conditioning_dim)
                inv_tau0 = 1.0 / (self.eps + self.init_temperature)
                eff_temperature = self.eps + 1.0 / (self.tau_fc(last_hidden_state) + inv_tau0).reshape(batch_size, -1, 1)
                if self.nbr_learnable_temperatures==1 \
                and self.nbr_learnable_temperatures != seq_len:
                    eff_temperature = eff_temperature.repeat(1, seq_len, 1)
        else:
            eff_temperature = torch.tensor([self.init_temperature], device=self.device)

        u = torch.rand_like(x) * (0.999 - self.eps) + self.eps
        gumbels = -torch.log(-torch.log(u))
        gumbel_logits = x + gumbels

        y_soft = F.softmax(gumbel_logits / eff_temperature, dim=-1)

        if self.stgs_hard:
            message_ids = torch.distributions.Categorical(probs=y_soft).sample()
            y_hard = F.one_hot(message_ids, num_classes=self.vocab_size)
            y_hard = y_hard.half() if x.dtype == torch.half else y_hard.float()
            message_one_hot = y_hard - y_soft.detach() + y_soft
        else:
            message_ids = torch.distributions.Categorical(probs=y_soft).sample()
            message_one_hot = y_soft

        message_one_hot = message_one_hot.half() if x.dtype == torch.half else message_one_hot.float()

        return message_ids, message_one_hot, eff_temperature


def stgs_forward_pass(
    *,
    model,
    tokenizer,
    loss_instance,
    learnable_inputs: torch.Tensor,
    stgs_module: STGS,
    embedding_weights_subset: torch.Tensor,
    allowed_tokens: torch.Tensor,
    target_length: int,
    batch_size: int,
    device: torch.device,
    model_precision: str,
    pre_prompt: Optional[str],
    bptt: bool,
    bptt_stgs: Optional[STGS],
    filter_vocab: bool,
) -> ForwardPassResult:
    message_logits = learnable_inputs.repeat(batch_size, 1, 1)
    if model_precision == "half":
        message_logits = message_logits.half()
    message_ids, message_one_hot, eff_temperature = stgs_module.forward(message_logits)

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
    hidden_states = outputs.hidden_states
    logits_allowed = logits[..., allowed_tokens]
    prompt_logits = logits_allowed
    past_key_values = outputs.past_key_values

    all_logits = [logits_allowed[:, -1:, :]]
    all_hidden_states = [outputs.hidden_states]

    completion_ids = []
    current_length = 1
    bptt_eff_temperature = torch.tensor(0.0, device=device)
    while current_length < target_length:
        if bptt:
            assert bptt_stgs is not None, "BPTT requested but no STGS module provided for it."
            next_token_id, next_token_one_hot, bptt_eff_temperature = bptt_stgs(
                all_logits[-1],
                hidden_states=outputs.hidden_states,
            )
            next_token_embedding = torch.matmul(next_token_one_hot, embedding_weights_subset)
        else:
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

    estimator_state: Dict[str, Optional[torch.Tensor]] = {
        "eff_temperature": eff_temperature,
        "bptt_eff_temperature": bptt_eff_temperature,
    }

    return ForwardPassResult(
        loss=loss,
        backward_loss=loss,
        losses_dict=losses_dict,
        generated_logits=generated_logits,
        prompt_ids=message_ids,
        prompt_logits=prompt_logits,
        completion_ids=completion_ids,
        estimator_state=estimator_state,
    )


def estimate_stgs_gradient_variance(
    num_samples: int,
    baseline_grad: Optional[torch.Tensor],
    forward_pass_fn,
    learnable_inputs: torch.Tensor,
) -> Dict[str, float]:
    if num_samples < 2 or baseline_grad is None:
        return {}

    grad_mean = baseline_grad.clone()
    grad_m2 = torch.zeros_like(grad_mean)
    sample_count = 1

    for _ in range(num_samples - 1):
        sample_result: ForwardPassResult = forward_pass_fn()
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
        "stgs_grad_variance_mean": variance.mean().item(),
        "stgs_grad_variance_max": variance.max().item(),
        "stgs_grad_variance_norm": variance.norm().item(),
        "stgs_grad_mean_norm": grad_mean.norm().item(),
        "stgs_grad_variance_samples": int(sample_count),
    }


def estimate_stgs_gradient_bias(
    *,
    stgs_num_samples: int,
    reinforce_num_samples: int,
    baseline_grad: Optional[torch.Tensor],
    stgs_forward_pass_fn: Callable[[], ForwardPassResult],
    reinforce_forward_pass_fn: Callable[..., ForwardPassResult],
    learnable_inputs: torch.Tensor,
    reinforce_update_baseline: bool,
) -> Dict[str, float]:
    if baseline_grad is None:
        return {}

    if stgs_num_samples < 1 or reinforce_num_samples < 1:
        return {}

    stgs_grad_mean = baseline_grad.clone()
    stgs_sample_count = 1

    for _ in range(stgs_num_samples - 1):
        sample_result = stgs_forward_pass_fn()
        sample_grad = torch.autograd.grad(
            sample_result.backward_loss,
            learnable_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].detach()

        stgs_sample_count += 1
        stgs_grad_mean += sample_grad

        del sample_result, sample_grad

    stgs_grad_mean = stgs_grad_mean / stgs_sample_count

    reinforce_grad_sum = torch.zeros_like(baseline_grad)
    reinforce_sample_count = 0

    for _ in range(reinforce_num_samples):
        sample_result = reinforce_forward_pass_fn(update_baseline=reinforce_update_baseline)
        sample_grad = torch.autograd.grad(
            sample_result.backward_loss,
            learnable_inputs,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0].detach()

        reinforce_sample_count += 1
        reinforce_grad_sum += sample_grad

        del sample_result, sample_grad

    if reinforce_sample_count < 1:
        return {}

    reinforce_grad_mean = reinforce_grad_sum / reinforce_sample_count
    bias_tensor = stgs_grad_mean - reinforce_grad_mean

    return {
        "stgs_grad_bias_norm": bias_tensor.norm().item(),
        "stgs_grad_bias_mean": bias_tensor.mean().item(),
        "stgs_grad_bias_abs_mean": bias_tensor.abs().mean().item(),
        "stgs_grad_bias_abs_max": bias_tensor.abs().max().item(),
        "stgs_grad_mean_norm": stgs_grad_mean.norm().item(),
        "stgs_grad_bias_reference_grad_mean_norm": reinforce_grad_mean.norm().item(),
        "stgs_grad_bias_stgs_samples": int(stgs_sample_count),
        "stgs_grad_bias_reference_samples": int(reinforce_sample_count),
    }
