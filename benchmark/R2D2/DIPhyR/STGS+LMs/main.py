import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
import argparse
import copy
import wandb
import matplotlib.pyplot as plt
from functools import partial
from typing import Dict

from gradient_estimators import (
    STGS,
    ReinforceEstimator,
    stgs_forward_pass,
    reinforce_forward_pass,
    estimate_stgs_gradient_variance,
    estimate_stgs_gradient_bias,
    estimate_reinforce_gradient_variance,
)


def setup_model_and_tokenizer(model_name="HuggingFaceM4/tiny-random-LlamaForCausalLM",device='cpu',model_precision='full'):
    """
    Load model and tokenizer
    """
    # Load model and move to device
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode (frozen)

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    model = model.to(device)
    if model_precision == "half":
        model.half()
    
    # Load tokenizer (only needed for the target sequence)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer

def initialize_learnable_inputs(vocab_size, seq_len, device, batch_size=1):
    """
    Initialize learnable input as one-hot encoded vectors
    """
    # Create learnable input parameters (random initialization)
    # Perform the multiplication *before* passing it to nn.Parameter.
    learnable_inputs = torch.randn((batch_size, seq_len, vocab_size), requires_grad=True, device=device)

    return learnable_inputs

def filter_vocabulary(embedding_weights, seed_vector, target_token_ids, threshold=0.5):
    """
    Compute cosine similarities between each vocabulary token embedding and seed_vector.
    Return the indices of tokens with similarity above the threshold,
    while always including the target_token_ids.
    """
    norm_weights = embedding_weights / (embedding_weights.norm(dim=1, keepdim=True) + 1e-9)
    norm_seed = seed_vector / (seed_vector.norm() + 1e-9)
    similarities = torch.matmul(norm_weights, norm_seed.unsqueeze(1)).squeeze(1)
    allowed = (similarities >= threshold).nonzero(as_tuple=True)[0]

    # Ensure target tokens are included
    target_set = set(target_token_ids.tolist())
    allowed_set = set(allowed.tolist())
    union_allowed = allowed_set.union(target_set)
    union_allowed = torch.tensor(sorted(list(union_allowed)), device=embedding_weights.device)
    return union_allowed

def generate_completions(model, input_embeddings, target_length, temperature=1.0, pre_prompt=None):
    """
    Generate completions from the model using the input embeddings
    """
    # Store the original inputs for later use
    batch_size = input_embeddings.shape[0]
    embedding_dim = input_embeddings.shape[2]

    # Initialize past_key_values as None
    past_key_values = None

    # Get the embedding layer for shape reference and token-to-embedding conversion
    embedding_layer = model.get_input_embeddings()

    # Process pre-prompt if provided
    if pre_prompt is not None:
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided when using pre_prompt")

        # Tokenize the pre-prompt
        pre_prompt_tokens = tokenizer(pre_prompt, return_tensors="pt").input_ids.to(input_embeddings.device)

        # Convert tokens to embeddings
        pre_prompt_embeds = embedding_layer(pre_prompt_tokens)

        # Combine pre-prompt embeddings with learnable embeddings
        # [batch_size, pre_prompt_length + learnable_length, hidden_size]
        combined_embeds = torch.cat([pre_prompt_embeds, input_embeddings], dim=1)
        current_embeds = combined_embeds
    else:
        # Use just the learnable embeddings
        current_embeds = input_embeddings

    # Store all generated token ids
    all_token_ids = []

    # Autoregressive generation loop
    for _ in range(target_length):
        # Forward pass
        outputs = model(
            inputs_embeds=current_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        # Get the logits for the next token
        next_token_logits = outputs.logits[:, -1, :]

        # Apply temperature
        next_token_logits = next_token_logits / temperature

        # Get the most likely next token
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        all_token_ids.append(next_token_id)

        # Convert token to embedding for next iteration
        next_token_embed = embedding_layer(next_token_id)

        # Set up for next iteration
        current_embeds = next_token_embed
        past_key_values = outputs.past_key_values

    # Concatenate all token ids
    generated_token_ids = torch.cat(all_token_ids, dim=1)

    return generated_token_ids, outputs.hidden_states

class LossClass(object):
    def __init__(
        self,
        losses,
        embedding_weights_subset,
        target_text,
        target_tokens_mapped,
        model,
        tokenizer,
        kwargs,
    ):
        self.eps = 1e-8
        self.losses = losses
        self.target_text = target_text
        self.target_tokens_mapped = target_tokens_mapped
        self.embedding_weights_subset = embedding_weights_subset
        self.model = model
        self.tokenizer = tokenizer

        self.vocab_size = self.embedding_weights_subset.shape[0]
        self.embedding_dim = self.embedding_weights_subset.shape[1]
        self.batch_size = self.target_tokens_mapped.shape[0]
        self.target_seq_len = self.target_tokens_mapped.shape[1]
        self.hidden_state_dim = self.model.config.hidden_size

        self.kwargs = kwargs

        # EmbXEntropy:
        if 'embxentropy' in self.losses.lower():
            target_tokens_one_hot = F.one_hot(
                self.target_tokens_mapped,
                num_classes=self.vocab_size,
            ).float()
            # batch_size x target_seq_len x vocab_size
            target_embeddings = torch.matmul(target_tokens_one_hot, self.embedding_weights_subset)
            # batch_size x target_seq_len x embedding_dim
            '''
            diff_target_minus_other = target_embeddings.unsqueeze(2).expand(-1,-1,self.vocab_size, -1) - self.embedding_weights_subset.reshape(
                1,1,self.vocab_size, self.embedding_dim,
            ).expand(
                self.batch_size, self.target_seq_len, -1,-1,
            )
            # batch_size x target_seq_len x vocab_size x embedding_dim
            l2_norm_diff_target_other = torch.linalg.norm(
                diff_target_minus_other,
                dim=-1,
                ord=2,
            )
            # batch_size x target_seq_len x vocab_size
            '''
            l2_norm_diff_target_other = torch.zeros((self.batch_size, self.target_seq_len, self.vocab_size))
            for bidx in range(self.batch_size):
                for tidx in range(self.target_seq_len):
                    diff_target_minus_other = target_embeddings[bidx,tidx].unsqueeze(0).expand(self.vocab_size, -1) - self.embedding_weights_subset
                    # (vocab_size x emd_dim)
                    l2_norm_diff_target_other[bidx,tidx] = torch.linalg.norm(
                        diff_target_minus_other,
                        dim=-1,
                        ord=2,
                    )
                    # vocab_size

            self.target_distr = torch.softmax(
                1.0 / (self.eps + l2_norm_diff_target_other),
                dim=-1,
            ).to(device=self.embedding_weights_subset.device)

        elif 'embLayer' in self.losses:
            target_tokens_one_hot = F.one_hot(
                self.target_tokens_mapped,
                num_classes=self.vocab_size,
            ).float()
            # batch_size x target_seq_len x vocab_size
            target_embeddings = torch.matmul(target_tokens_one_hot, self.embedding_weights_subset)
            # batch_size x target_seq_len x embedding_dim
            l2_norm_diff_target_other = torch.zeros((self.batch_size, self.target_seq_len, self.vocab_size))
            
            target_outputs = self.model(
                inputs_embeds=target_embeddings,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            
            if 'L2' in self.losses:
                loss_type = 'L2'
            elif 'cos' in self.losses.lower():
                loss_type = 'Cos'

            self.embedding_layers = [int(number) for number in self.losses.split('embLayer')[1].split(loss_type)[0].split('+')]
            
            self.target_embeddings = { emblayer:target_outputs.hidden_states[emblayer].to(device=self.embedding_weights_subset.device) 
                for emblayer in self.embedding_layers
            }
            # (batch_size x target_seq_len x hidden_state)

    def compute_perplexity(
        self,
        all_logits,
        predictions,
    ):
        '''
        Compute perplexity with log:
        :param all_logits: batch_size x seq_len x vocab_size
        :param predictions: batch_size x seq_len 
        '''
        lslhd = all_logits 
        #(batch_size x seq_len x vocab_size)
        batch_size = all_logits.shape[0]
        seq_len = all_logits.shape[1]
        vocab_size = all_logits.shape[2]

        lslhd = lslhd.log_softmax(dim=-1)
        lslhd = lslhd.gather(
            dim=-1, 
            index=predictions.unsqueeze(-1),
        ).squeeze(-1)
        # (batch_size x seq_len)
        #lslhd = lsoftmaxed_pl.gather(dim=-1, index=tokenized_prediction.unsqueeze(-1)).squeeze(-1)
        if self.tokenizer.pad_token_id is not None:
            lnotpadding_mask = (predictions != self.tokenizer.pad_token_id).float()
            #lnotpadding_mask = (tokenized_prediction != self.tokenizer.pad_token_id).float()
            #options_true_length = (batched_options_inputs.input_ids != self.tokenizer.pad_token_id).long().sum(dim=-1).unsqueeze(-1)
            #(batch_size x 1)
            lslhd = lnotpadding_mask * lslhd
        else:
            lnotpadding_mask = torch.ones_like(predictions)
        #torch.pow(slhd, 1.0/options_true_length)
        #(option_batch_size x option_len)
        #print('cache option: ', lslhd.shape)
        #print(lslhd)
        lsentences_likelihoods = lslhd.sum(dim=-1) #= slhd.cpu().prod(dim=-1).to(slhd.device)
        #(batch_size )
        lsentences_perplexities = torch.exp(-lsentences_likelihoods / (lnotpadding_mask.sum(dim=-1)+1e-8)) #1.0/(slhd+1e-8)
        # (batch_size )
        
        return lsentences_perplexities

    def compute_loss(
        self,
        input_dict,
    ):
        sumloss = 0
        losses_dict = {}

        if "crossentropy" in self.losses.lower():
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                #generated_logits.reshape(-1, vocab_size),
                input_dict['generated_logits'].reshape(-1, self.embedding_weights_subset.shape[0]),
                self.target_tokens_mapped.reshape(-1), #.reshape(1, -1).repeat(batch_size, 1),
                #target_tokens.reshape(-1)
            )
            losses_dict['crossentropy'] = loss
            sumloss += loss

        if "embedded" in self.losses.lower():
            #print(self.embedding_weights_subset.shape)
            #print(input_dict['generated_logits'].shape)
            generated_distr = input_dict['generated_logits'].softmax(dim=-1)
            generated_embeddings = torch.matmul( 
                generated_distr,
                self.embedding_weights_subset,
            )
            #print(self.target_tokens_mapped.shape)
            target_tokens_one_hot = F.one_hot(self.target_tokens_mapped, num_classes=self.embedding_weights_subset.shape[0]).float()
            #print(target_tokens_one_hot.shape)
            target_embeddings = torch.matmul(target_tokens_one_hot, self.embedding_weights_subset)
            loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')#'mean')
            loss = loss_fn(
                input=generated_embeddings,
                target=target_embeddings.detach(),
            )
            # (batch_size x target_seq_len x embedding_size)
            #loss = loss.mean() #dim=-1).mean(dim=-1)
            #loss = loss.sum(dim=-1).sqrt().mean()
            loss = loss.sum(dim=-1).mean()
            # (batch_size
            losses_dict['embedded'] = loss
            sumloss += loss

        if "embxentropy" in self.losses.lower():
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                input_dict['generated_logits'].reshape(-1, self.vocab_size),
                target=self.target_distr.reshape(-1, self.vocab_size).detach(),
            )
            losses_dict['embxentropy'] = loss
            sumloss += loss

        if 'embLayer' in self.losses :
            generated_embeddings = {}
            for emblayer in self.embedding_layers:
                generated_embeddings[emblayer] = [hs[emblayer][:,-1:] for hs in input_dict['generated_hidden_states']]
                generated_embeddings[emblayer] = torch.cat(generated_embeddings[emblayer], dim=1)
                # (batch_size x target_seq_len x hidden-dim)

            losses = {}
            if 'L2' in self.losses:
                loss_type = 'L2'
                loss_fn = torch.nn.MSELoss(size_average=None, reduce=None, reduction='none')#'mean')
                for emblayer in self.embedding_layers:
                    losses[emblayer] = loss_fn(
                        input=generated_embeddings[emblayer],
                        target=self.target_embeddings[emblayer].detach(),
                    )
                    # (batch_size x target_seq_len x embedding_size)
                    #loss = loss.mean() #dim=-1).mean(dim=-1)
                    #loss = loss.sum(dim=-1).sqrt().mean()
                    losses[emblayer] = losses[emblayer].sum(dim=-1).mean()
                    # (batch_size
                    losses_dict[f"embLayer{emblayer}L2"] = losses[emblayer]
            elif 'cos' in self.losses.lower():
                loss_type = 'Cos'
                loss_fn = torch.nn.CosineEmbeddingLoss(reduction='mean')
                target = torch.ones((self.batch_size*self.target_seq_len,), device=self.embedding_weights_subset.device) 
                for emblayer in self.embedding_layers:
                    losses[emblayer] = loss_fn(
                        input1=generated_embeddings[emblayer].reshape(-1,self.hidden_state_dim),
                        input2=self.target_embeddings[emblayer].detach().reshape(-1, self.hidden_state_dim),
                        target=target,
                    )
                    # scalere because mean over (batch_size * target_seq_len) ?
                    losses_dict[f"embLayer{emblayer}{loss_type}"] = losses[emblayer]
            loss = sum(losses.values())
            losses_dict[f"embLayer{loss_type}"] = loss
            sumloss += loss

        if 'perplexity' in self.losses.lower():
            if 'promptPerplexity' in self.losses:
                promptPLX = self.compute_perplexity(
                    all_logits=input_dict['prompt_logits'],
                    predictions=input_dict['prompt_ids'],
                )
            else:
                promptPLX = torch.zeros(self.batch_size)

            if 'completionPerplexity' in self.losses:
                complPLX = self.compute_perplexity(
                    all_logits=input_dict['generated_logits'],
                    predictions=input_dict['completion_ids'],
                )
            else:
                complPLX = torch.zeros(self.batch_size)
            
            promptLambda = self.kwargs['promptLambda']
            complLambda = self.kwargs['complLambda']
            loss = (complLambda*complPLX + promptLambda*promptPLX).mean()
            losses_dict[f"PLX-prompt"] = promptPLX.mean()
            losses_dict[f"PLX-completion"] = complPLX.mean()
            sumloss += loss

        losses_dict['sumloss'] = sumloss

        return losses_dict

class TokenOverlapMetric(object):
    def __init__(
        self,
        target_text,
        tokenizer,
    ):
        self.target_text = target_text
        self.tokenizer = tokenizer

    def measure(
        self,
        prompt_text=None,
        prompt_tokens=None,
    ):
        output_dict = {}

        target_tokens = self.tokenizer(
            self.target_text, 
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        if prompt_tokens is None:
            assert prompt_text is not None
            prompt_tokens = self.tokenizer(
                prompt_text, 
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]
        elif isinstance(prompt_tokens, list):
            prompt_tokens = torch.Tensor(prompt_tokens) 
        
        tt_occ = {}
        for ttoken in target_tokens:
            tt_occ[ttoken.item()] = (prompt_tokens == ttoken).int().sum().item()

        nbr_occ = sum(tt_occ.values())
        max_occ = prompt_tokens.shape[-1]

        overlap_ratio = nbr_occ / max_occ
        output_dict['token_overlap_ratio'] = overlap_ratio

        target_set = set(target_tokens.tolist())
        target_set_size = len(target_set)
        target_hits = {
            ttoken: int(t_occ>0)
            for ttoken, t_occ in tt_occ.items()
        }
        target_hits_size = sum(target_hits.values())
        output_dict['target_hit_ratio'] = float(target_hits_size) / target_set_size
        
        return output_dict


def optimize_inputs(
    model,
    tokenizer,
    device,
    losses="crossentropy",
    bptt=False,
    target_text="The quick brown fox jumps over the lazy dog",
    pre_prompt=None,
    seq_len=50,
    epochs=1000,
    learning_rate=0.01,
    temperature = 0.5,
    bptt_temperature = 0.5,
    learnable_temperature=False,
    bptt_learnable_temperature=False,
    stgs_hard=True,
    bptt_stgs_hard=True,
    bptt_hidden_state_conditioning=False,
    plot_every=10,
    log_table_every=100,
    stgs_grad_variance_samples=0,
    stgs_grad_variance_period=1,
    stgs_grad_bias_samples=0,
    stgs_grad_bias_period=1,
    stgs_grad_bias_reference_samples=0,
    stgs_grad_bias_reference_batch_size=None,
    stgs_grad_bias_reference_use_baseline=True,
    stgs_grad_bias_reference_reward_scale=1.0,
    stgs_grad_bias_reference_baseline_beta=0.9,
    gradient_estimator="stgs",
    reinforce_grad_variance_samples=0,
    reinforce_grad_variance_period=1,
    reinforce_reward_scale=1.0,
    reinforce_use_baseline=True,
    reinforce_baseline_beta=0.9,
    eps=1e-10,
    bptt_eps=1e-10,
    vocab_threshold=0.5,  # Hyperparameter for filtering
    filter_vocab=False,
    max_gradient_norm=0.0,
    batch_size=1,
    kwargs={},
):
    """
    Optimize input embeddings to make the frozen model produce the target output as a completion
    """

    # Enabling Gradient checkpointing:
    if kwargs.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    # Get model's vocabulary size and embedding dimension
    vocab_size = model.config.vocab_size
    hidden_state_dim = model.config.hidden_size

    # Tokenize the target text
    target_tokens = tokenizer(target_text, return_tensors="pt").input_ids.to(device)  # shape: (1, target_length)
    target_length = target_tokens.shape[1]

    # Get embedding matrix from the model
    embedding_layer = model.get_input_embeddings()
    full_embedding_weights = embedding_layer.weight.detach()  # (vocab_size, embed_dim)

    # Compute a seed vector from the target text (average embedding)
    if filter_vocab:
      target_embeds = embedding_layer(target_tokens)  # (1, target_length, embed_dim)
      seed_vector = target_embeds.mean(dim=1).squeeze(0)  # (embed_dim,)

      # Filter full vocabulary based on cosine similarity and ensure target tokens are included
      allowed_tokens = filter_vocabulary(full_embedding_weights, seed_vector, target_tokens[0], threshold=vocab_threshold)
      allowed_vocab_size = allowed_tokens.shape[0]
    else:
      allowed_tokens = torch.arange(vocab_size, device=device)
      allowed_vocab_size = vocab_size
    print(f"Restricted vocabulary size: {allowed_vocab_size} tokens (from full vocab size {full_embedding_weights.shape[0]})")

    # W&B log a table of the allowed tokens and the target_text tokens:
    allowed_tokens_list = allowed_tokens.tolist()
    target_tokens_list = target_tokens[0].tolist()
    bias_reference_batch_size_value = (
        stgs_grad_bias_reference_batch_size
        if stgs_grad_bias_reference_batch_size and stgs_grad_bias_reference_batch_size > 0
        else batch_size
    )

    wandb.log({
        "allowed_tokens": allowed_tokens_list,
        "target_tokens": target_tokens_list,
        "allowed_tokens_str": [tokenizer.decode(t) for t in allowed_tokens_list],
        "target_tokens_str": [tokenizer.decode(t) for t in target_tokens_list],
        "allowed_vocab_size": allowed_vocab_size,
        "target_text": target_text,
        "pre_prompt": pre_prompt,
        "seq_len": seq_len,
        "epochs": epochs,
        "losses": losses,
        "learning_rate": learning_rate,
        "bptt":bptt,
        "temperature": temperature,
        "bptt_temperature": bptt_temperature,
        "learnable_temperature": learnable_temperature,
        "bptt_learnable_temperature": bptt_learnable_temperature,
        "stgs_hard": stgs_hard,
        "bptt_stgs_hard": bptt_stgs_hard,
        "bptt_hidden_state_conditioning":bptt_hidden_state_conditioning,
        "plot_every": plot_every,
        "eps": eps,
        "bptt_eps": bptt_eps,
        "vocab_threshold": vocab_threshold,
        "batch_size": batch_size,
        "stgs_grad_variance_samples_cfg": stgs_grad_variance_samples,
        "stgs_grad_variance_period_cfg": stgs_grad_variance_period,
        "reinforce_grad_variance_samples_cfg": reinforce_grad_variance_samples,
        "reinforce_grad_variance_period_cfg": reinforce_grad_variance_period,
        "stgs_grad_bias_samples_cfg": stgs_grad_bias_samples,
        "stgs_grad_bias_period_cfg": stgs_grad_bias_period,
        "stgs_grad_bias_reference_samples_cfg": stgs_grad_bias_reference_samples,
        "stgs_grad_bias_reference_batch_size_cfg": bias_reference_batch_size_value,
        "stgs_grad_bias_reference_use_baseline_cfg": stgs_grad_bias_reference_use_baseline,
        "stgs_grad_bias_reference_reward_scale_cfg": stgs_grad_bias_reference_reward_scale,
        "stgs_grad_bias_reference_baseline_beta_cfg": stgs_grad_bias_reference_baseline_beta,
        "gradient_estimator": gradient_estimator,
        "reinforce_reward_scale": reinforce_reward_scale,
        "reinforce_use_baseline": reinforce_use_baseline,
        "reinforce_baseline_beta": reinforce_baseline_beta,
    })
    wandb_table = wandb.Table(columns=[
        "epoch", 
        "target_output_str", 
        "learned_input_ids", 
        "learned_input_str", 
        "generated_output_ids", 
        "generated_output_str",
        "token_overlap_ratio",
        "target_hit_ratio",
    ])

    if filter_vocab:
      # Build a mapping from full-vocab token id to allowed index
      allowed_list = allowed_tokens.tolist()
      mapping = { token_id: idx for idx, token_id in enumerate(allowed_list) }

      # Remap target tokens to allowed vocabulary indices using a list comprehension
      target_tokens_mapped = torch.tensor([mapping[t.item()] for t in target_tokens[0]], device=device).unsqueeze(0)

      # Get the subset of the embedding matrix corresponding to allowed tokens
      embedding_weights_subset = full_embedding_weights[allowed_tokens]  # (allowed_vocab_size, embed_dim)
    else:
      target_tokens_mapped = target_tokens
      embedding_weights_subset = full_embedding_weights
    target_tokens_mapped = target_tokens_mapped.reshape(1, -1).repeat(batch_size, 1)
    # Check shape:
    #print(f"Target tokens mapped shape: {target_tokens_mapped.shape}")

    # Initialize learnable inputs
    parameters = []
    learnable_inputs = initialize_learnable_inputs(allowed_vocab_size, seq_len, device)
    parameters.append(learnable_inputs)
    
    token_overlap_metric = TokenOverlapMetric(
        target_text=target_text,
        tokenizer=tokenizer,
    )

    # Set up loss function
    loss_instance = LossClass(
        model=model,
        tokenizer=tokenizer,
        embedding_weights_subset=embedding_weights_subset,
        losses=losses,
        target_text=target_text,
        target_tokens_mapped=target_tokens_mapped,
        kwargs=kwargs,
    )

    estimator_choice = gradient_estimator.lower()
    stgs_module = None
    bptt_stgs_module = None
    reinforce_helper = None

    if estimator_choice == "stgs":
        stgs_module = STGS(
            vocab_size=allowed_vocab_size,
            stgs_hard=stgs_hard,
            init_temperature=temperature,
            learnable_temperature=learnable_temperature,
            eps=eps,
            device=device,
        )
        parameters += list(stgs_module.parameters())

        if bptt:
            bptt_stgs_module = STGS(
                vocab_size=allowed_vocab_size,
                stgs_hard=bptt_stgs_hard,
                init_temperature=bptt_temperature,
                learnable_temperature=bptt_learnable_temperature,
                eps=bptt_eps,
                conditioning_dim=hidden_state_dim if bptt_hidden_state_conditioning else 0,
                device=device,
            )
            parameters += list(bptt_stgs_module.parameters())
    elif estimator_choice == "reinforce":
        if bptt:
            raise ValueError("BPTT is currently not supported with the REINFORCE estimator.")
        reinforce_helper = ReinforceEstimator(
            reward_scale=kwargs.get("reinforce_reward_scale", reinforce_reward_scale),
            use_baseline=kwargs.get("reinforce_use_baseline", reinforce_use_baseline),
            baseline_beta=kwargs.get("reinforce_baseline_beta", reinforce_baseline_beta),
        )
    else:
        raise ValueError(f"Unknown gradient estimator '{gradient_estimator}'.")

    optimizer = optim.Adam(parameters, lr=learning_rate)

    if estimator_choice == "stgs":
        forward_pass_callable = partial(
            stgs_forward_pass,
            model=model,
            tokenizer=tokenizer,
            loss_instance=loss_instance,
            learnable_inputs=learnable_inputs,
            stgs_module=stgs_module,
            embedding_weights_subset=embedding_weights_subset,
            allowed_tokens=allowed_tokens,
            target_length=target_length,
            batch_size=batch_size,
            device=device,
            model_precision=kwargs.get("model_precision", "full"),
            pre_prompt=pre_prompt,
            bptt=bptt,
            bptt_stgs=bptt_stgs_module,
            filter_vocab=filter_vocab,
        )
    else:
        forward_pass_callable = partial(
            reinforce_forward_pass,
            model=model,
            tokenizer=tokenizer,
            loss_instance=loss_instance,
            learnable_inputs=learnable_inputs,
            embedding_weights_subset=embedding_weights_subset,
            allowed_tokens=allowed_tokens,
            target_length=target_length,
            batch_size=batch_size,
            device=device,
            model_precision=kwargs.get("model_precision", "full"),
            pre_prompt=pre_prompt,
            filter_vocab=filter_vocab,
            reinforce_helper=reinforce_helper,
        )

    estimator_is_stgs = estimator_choice == "stgs"

    # Training loop
    losses = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        step_result = forward_pass_callable()
        loss = step_result.loss
        backward_loss = step_result.backward_loss
        losses_dict = step_result.losses_dict
        generated_logits = step_result.generated_logits
        estimator_state = step_result.estimator_state

        backward_loss.backward()

        estimator_metrics: Dict[str, float] = {}
        if estimator_is_stgs:
            measure_grad_variance = (
                stgs_grad_variance_samples >= 2
                and stgs_grad_variance_period > 0
                and (epoch % stgs_grad_variance_period == 0)
            )
            measure_grad_bias = (
                stgs_grad_bias_samples >= 1
                and stgs_grad_bias_reference_samples >= 1
                and stgs_grad_bias_period > 0
                and (epoch % stgs_grad_bias_period == 0)
            )

            need_baseline_grad = (
                (measure_grad_variance or measure_grad_bias)
                and learnable_inputs.grad is not None
            )
            baseline_grad = None
            if need_baseline_grad:
                baseline_grad = learnable_inputs.grad.detach().clone()

            if measure_grad_variance and baseline_grad is not None:
                estimator_metrics.update(
                    estimate_stgs_gradient_variance(
                        stgs_grad_variance_samples,
                        baseline_grad,
                        forward_pass_callable,
                        learnable_inputs,
                    )
                )

            if measure_grad_bias and baseline_grad is not None:
                bias_reinforce_helper = ReinforceEstimator(
                    reward_scale=stgs_grad_bias_reference_reward_scale,
                    use_baseline=stgs_grad_bias_reference_use_baseline,
                    baseline_beta=stgs_grad_bias_reference_baseline_beta,
                )
                bias_reinforce_forward = partial(
                    reinforce_forward_pass,
                    model=model,
                    tokenizer=tokenizer,
                    loss_instance=loss_instance,
                    learnable_inputs=learnable_inputs,
                    embedding_weights_subset=embedding_weights_subset,
                    allowed_tokens=allowed_tokens,
                    target_length=target_length,
                    batch_size=bias_reference_batch_size_value,
                    device=device,
                    model_precision=kwargs.get("model_precision", "full"),
                    pre_prompt=pre_prompt,
                    filter_vocab=filter_vocab,
                    reinforce_helper=bias_reinforce_helper,
                )

                bias_metrics = estimate_stgs_gradient_bias(
                    stgs_num_samples=stgs_grad_bias_samples,
                    reinforce_num_samples=stgs_grad_bias_reference_samples,
                    baseline_grad=baseline_grad,
                    stgs_forward_pass_fn=forward_pass_callable,
                    reinforce_forward_pass_fn=bias_reinforce_forward,
                    learnable_inputs=learnable_inputs,
                    reinforce_update_baseline=stgs_grad_bias_reference_use_baseline,
                )
                estimator_metrics.update(bias_metrics)
        else:
            measure_grad_variance = (
                reinforce_grad_variance_samples >= 2
                and reinforce_grad_variance_period > 0
                and (epoch % reinforce_grad_variance_period == 0)
            )
            baseline_grad = None
            if measure_grad_variance and learnable_inputs.grad is not None:
                baseline_grad = learnable_inputs.grad.detach().clone()
            if measure_grad_variance and baseline_grad is not None:
                estimator_metrics.update(
                    estimate_reinforce_gradient_variance(
                        reinforce_grad_variance_samples,
                        baseline_grad,
                        forward_pass_callable,
                        learnable_inputs,
                    )
                )

        if max_gradient_norm != 0.0:
            torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)

        optimizer.step()

        grad_tensor = learnable_inputs.grad
        grad_norm = grad_tensor.norm().item() if grad_tensor is not None else 0.0
        non_zero_grads = int((grad_tensor != 0).sum().item()) if grad_tensor is not None else 0
        grad_mean = grad_tensor.mean().item() if grad_tensor is not None else 0.0
        grad_max = grad_tensor.max().item() if grad_tensor is not None else 0.0
        grad_std = grad_tensor.std().item() if grad_tensor is not None else 0.0
        if grad_tensor is not None:
            grad_abs = grad_tensor.abs()
            grad_min = grad_abs[grad_abs > 0].min().item() if (grad_abs > 0).any() else 0.0
        else:
            grad_min = 0.0

        eff_temperature = estimator_state.get("eff_temperature")
        bptt_eff_temperature = estimator_state.get("bptt_eff_temperature")
        reinforce_baseline_state = estimator_state.get("reinforce_baseline")

        info_components = [f"Gradient norm: {grad_norm:.6f}", f"Allowed vocab size: {allowed_vocab_size}"]
        if torch.is_tensor(eff_temperature):
            info_components.insert(1, f"Tau = {eff_temperature.mean().item():.6f}")
        if reinforce_baseline_state is not None and torch.is_tensor(reinforce_baseline_state):
            info_components.append(f"Baseline: {reinforce_baseline_state.item():.6f}")
        info = " / ".join(info_components)

        losses.append(loss.item())
        pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f} / {info}")

        wandb_log = {
            "epoch": epoch + 1,
            "loss": loss.item(),
            "allowed_vocab_size": allowed_vocab_size,
            "non_zero_grads": non_zero_grads,
            "grad_mean": grad_mean,
            "grad_max": grad_max,
            "grad_min": grad_min,
            "grad_norm": grad_norm,
            "grad_std": grad_std,
            "vocab_size": model.config.vocab_size,
        }

        for k, v in losses_dict.items():
            wandb_log[k] = v.item()

        if estimator_is_stgs:
            if torch.is_tensor(eff_temperature):
                wandb_log["effective_temperature"] = eff_temperature.mean().item()
            if torch.is_tensor(bptt_eff_temperature):
                wandb_log["bptt_effective_temperature"] = bptt_eff_temperature.mean().item()
            elif isinstance(bptt_eff_temperature, (float, int)):
                wandb_log["bptt_effective_temperature"] = float(bptt_eff_temperature)
            wandb_log.update(estimator_metrics)
        else:
            advantage_val = estimator_state.get("reinforce_advantage")
            if torch.is_tensor(advantage_val):
                wandb_log["reinforce_advantage"] = advantage_val.item()
            baseline_val = estimator_state.get("reinforce_baseline")
            if torch.is_tensor(baseline_val):
                wandb_log["reinforce_baseline"] = baseline_val.item()
            log_prob_mean = estimator_state.get("reinforce_log_prob_mean")
            if torch.is_tensor(log_prob_mean):
                wandb_log["reinforce_log_prob_mean"] = log_prob_mean.item()
            log_prob_sum_mean = estimator_state.get("reinforce_log_prob_sum_mean")
            if torch.is_tensor(log_prob_sum_mean):
                wandb_log["reinforce_log_prob_sum_mean"] = log_prob_sum_mean.item()
            wandb_log.update(estimator_metrics)

        # Update wandb_table with generated_output:
        learnable_input_ids = torch.argmax(learnable_inputs, dim=-1)[0]
        generated_output_ids = torch.argmax(generated_logits, dim=-1)
        table_generated_output_ids = torch.gather(allowed_tokens.unsqueeze(0), dim=1, index=generated_output_ids[0:1])
        learnable_input_str = tokenizer.decode(learnable_input_ids, skip_special_tokens=False)
        generated_output_str = tokenizer.decode(table_generated_output_ids[0], skip_special_tokens=False)

        generated_tokens = table_generated_output_ids[0].cpu().tolist()

        metrics_dict = token_overlap_metric.measure(
            prompt_tokens=learnable_input_ids,
        )
        for k, v in metrics_dict.items():
            wandb_log[k] = v

        wandb_table.add_data(
            epoch+1, 
            target_text,
            learnable_input_ids.tolist(),
            learnable_input_str,
            table_generated_output_ids[0].tolist(),
            generated_output_str,
            *metrics_dict.values(),
        )

        wandb.log(wandb_log)
        
        if epoch % log_table_every == 0:
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})

         # Update plot every plot_every epochs - Colab compatible version
        if epoch % plot_every == 0 or epoch == epochs - 1:
            # Clear the output and create a new plot each time
            #clear_output(wait=True)

            # Create a new figure
            plt.figure(figsize=(10, 5))
            plt.plot(losses)
            plt.title('Training Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)

            # Save and display the plot
            plt.savefig('loss_curve_latest.png')
            #display(plt.gcf())  # This shows the plot in Colab

        if epoch > 0 and epoch % (plot_every * 5) == 0:
            with torch.no_grad():
                curr_input_embeddings = torch.matmul(learnable_inputs, embedding_weights_subset)
                generated_tokens, _ = generate_completions(
                    model,
                    curr_input_embeddings,
                    target_length=target_length,
                    pre_prompt=pre_prompt,
                )
                intermediate_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                print(f"\nEpoch {epoch} intermediate output: {intermediate_text}")
                print(f"Target: {target_text}")

        # Optional: early stopping condition
        if loss.item() < 0.01 \
        or generated_output_str == target_text:
            print(f"Converged at epoch {epoch+1} with loss: {loss.item():.6f}")
            wandb.log({"generated_output_table": copy.deepcopy(wandb_table)})
            break

    return generated_tokens, learnable_inputs, losses


def str2bool(instr):
    if isinstance(instr, bool):
        return instr
    if isinstance(instr, str):
        instr = instr.lower()
        if 'true' in instr:
            return True
        elif 'false' in instr:
            return False
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

def intOrNone(instr):
    if instr is None:
        return None
    return int(instr)

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('PromptOptimisationSTGSBenchmark')

    parser = argparse.ArgumentParser(description="Prompt Optimisation - STGS - Test.")
 
    #model_name = "HuggingFaceM4/tiny-random-LlamaForCausalLM"  # For example, using a small LLaMA model
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--model_precision", type=str, default="full")
    # full
    # half
    parser.add_argument("--gradient_checkpointing", type=str2bool, default=False)
    parser.add_argument("--target_text", type=str, default="The quick brown fox jumps over the lazy dog")
    parser.add_argument("--pre_prompt", type=str, default=None)
    #pre_prompt = "Complete the following: "  # Can be None if not needed
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=2000)
    #parser.add_argument("--losses", type=str, default="crossentropy")
    #parser.add_argument("--losses", type=str, default="embedded")
    parser.add_argument("--losses", type=str, default="embxentropy")
    # embLayerxL2
    # +embedded
    # +embxentropy
    # +perplexityPenalty
    parser.add_argument("--promptLambda", type=float, default=0.0)
    parser.add_argument("--complLambda", type=float, default=0.0)
    parser.add_argument("--learning_rate", type=float, default=1e-1)
    parser.add_argument("--max_gradient_norm", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-10)
    parser.add_argument("--bptt_eps", type=float, default=1e-10)
    parser.add_argument("--temperature", type=float, default=1e1)
    parser.add_argument("--learnable_temperature", type=str2bool, default=False)
    parser.add_argument("--stgs_hard", type=str2bool, default=False)
    parser.add_argument("--bptt", type=str2bool, default=False)
    parser.add_argument("--bptt_temperature", type=float, default=1e1)
    parser.add_argument("--bptt_learnable_temperature", type=str2bool, default=False)
    parser.add_argument("--bptt_stgs_hard", type=str2bool, default=False)
    parser.add_argument("--bptt_hidden_state_conditioning", type=str2bool, default=False)
    parser.add_argument("--plot_every", type=int, default=100000)
    parser.add_argument("--stgs_grad_variance_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_variance_period", type=int, default=1)
    parser.add_argument("--stgs_grad_bias_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_period", type=int, default=1)
    parser.add_argument("--stgs_grad_bias_reference_samples", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_reference_batch_size", type=int, default=0)
    parser.add_argument("--stgs_grad_bias_reference_use_baseline", type=str2bool, default=True)
    parser.add_argument("--stgs_grad_bias_reference_reward_scale", type=float, default=1.0)
    parser.add_argument("--stgs_grad_bias_reference_baseline_beta", type=float, default=0.9)
    parser.add_argument("--reinforce_grad_variance_samples", type=int, default=0)
    parser.add_argument("--reinforce_grad_variance_period", type=int, default=1)
    parser.add_argument("--gradient_estimator", type=str, default="stgs", choices=["stgs", "reinforce"])
    parser.add_argument("--reinforce_reward_scale", type=float, default=1.0)
    parser.add_argument("--reinforce_use_baseline", type=str2bool, default=True)
    parser.add_argument("--reinforce_baseline_beta", type=float, default=0.9)
    parser.add_argument("--filter_vocab", type=str2bool, default= True)
    parser.add_argument("--vocab_threshold", type=float, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    
    args = parser.parse_args()
    config = vars(args)
   
    if config['promptLambda'] > 0.0:    config['losses'] += '+promptPerplexity'
    if config['complLambda'] > 0.0:    config['losses'] += '+completionPerplexity'

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # Load model and tokenizer
    print(f"Loading model: {config['model_name']}")
    model, tokenizer = setup_model_and_tokenizer(config['model_name'],device=device, model_precision=config['model_precision'])

    config['vocab_size'] = model.config.vocab_size
    config['hidden_size'] = model.config.hidden_size
    
    wandb_run = wandb.init(project="prompt-optimization", config=config)
    
    torch.manual_seed(config["seed"])
    # Optimize inputs
    print(f"Starting optimization with target: {args.target_text}")
    generated_tokens, optimized_inputs, losses = optimize_inputs(
        model,
        tokenizer,
        losses=config['losses'],
        bptt=config['bptt'],
        device=device,
        target_text=config['target_text'],
        pre_prompt=config['pre_prompt'],
        seq_len=config['seq_len'],
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        temperature=config['temperature'],
        bptt_temperature=config['bptt_temperature'],
        bptt_learnable_temperature=config['bptt_learnable_temperature'],
        learnable_temperature=config['learnable_temperature'],
        stgs_hard=config['stgs_hard'],
        bptt_stgs_hard=config['bptt_stgs_hard'],
        bptt_hidden_state_conditioning=config['bptt_hidden_state_conditioning'],
        plot_every=config['plot_every'],
        stgs_grad_variance_samples=config['stgs_grad_variance_samples'],
        stgs_grad_variance_period=config['stgs_grad_variance_period'],
        stgs_grad_bias_samples=config['stgs_grad_bias_samples'],
        stgs_grad_bias_period=config['stgs_grad_bias_period'],
        stgs_grad_bias_reference_samples=config['stgs_grad_bias_reference_samples'],
        stgs_grad_bias_reference_batch_size=config['stgs_grad_bias_reference_batch_size'],
        stgs_grad_bias_reference_use_baseline=config['stgs_grad_bias_reference_use_baseline'],
        stgs_grad_bias_reference_reward_scale=config['stgs_grad_bias_reference_reward_scale'],
        stgs_grad_bias_reference_baseline_beta=config['stgs_grad_bias_reference_baseline_beta'],
        reinforce_grad_variance_samples=config['reinforce_grad_variance_samples'],
        reinforce_grad_variance_period=config['reinforce_grad_variance_period'],
        gradient_estimator=config['gradient_estimator'],
        reinforce_reward_scale=config['reinforce_reward_scale'],
        reinforce_use_baseline=config['reinforce_use_baseline'],
        reinforce_baseline_beta=config['reinforce_baseline_beta'],
        eps=config['eps'],
        bptt_eps=config['bptt_eps'],
        vocab_threshold=config['vocab_threshold'],
        filter_vocab=config['filter_vocab'],
        max_gradient_norm=config['max_gradient_norm'],
        batch_size=config['batch_size'],
        kwargs=config,
    )

    wandb_run.finish()


if __name__ == '__main__':
    main()
