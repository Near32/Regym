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

class STGS(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        stgs_hard=False,
        init_temperature=1.0,
        learnable_temperature=False,
        conditioning_dim=0,
        eps=1e-12,
        device="cpu",
    ):
        super(STGS,self).__init__()
        self.vocab_size = vocab_size
        self.stgs_hard = stgs_hard
        self.init_temperature = init_temperature
        self.learnable_temperature = learnable_temperature
        self.conditioning_dim = conditioning_dim
        self.eps = eps
        self.device = device

        if self.learnable_temperature:
            #self.register_parameter(name="temperature_param", param=torch.nn.Parameter(torch.rand(1, requires_grad=True, device=self.device)))
            if self.conditioning_dim < 1:
                self.temperature_param = torch.nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))
            else:
                self.tau_fc = nn.Sequential(
                    nn.Linear(self.conditioning_dim, 1,bias=False),
                    nn.Softplus()
                )
                self.tau_fc = self.tau_fc.to(device=device)

    def forward(self, x, hidden_states=None):
        if self.learnable_temperature:
            if self.conditioning_dim < 1:
                eff_temperature = self.eps + 1. / (F.softplus(self.temperature_param)+1.0/(self.eps+self.init_temperature))
            else:
                assert hidden_states is not None
                batch_size = x.shape[0]
                seq_len = x.shape[1]
                last_hidden_state = hidden_states[-1][:,-1,:].reshape(batch_size, self.conditioning_dim)
                self.inv_tau0 = 1.0/(self.eps+self.init_temperature)
                eff_temperature = self.eps + 1. / ( self.tau_fc(last_hidden_state)+self.inv_tau0).reshape(batch_size, -1, 1)
                # repeat for seq len:
                eff_temperature = eff_temperature.repeat(1,seq_len,1)
        else:
          eff_temperature = torch.tensor([self.init_temperature], device=self.device)
        
        # Add Gumbel noise for exploration during training
        '''
        gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.zeros_like(message_logits),
            torch.ones_like(message_logits)
        )

        gss = []
        for bidx in range(batch_size):
          gumbel_sample = gumbel_dist.sample()
          # (1, allowed_vocab, seq_len)
          #print(gumbel_sample.shape)
          gss.append(gumbel_sample)
        gumbel_sample = torch.concat(gss, dim=0)
        #gumbel_logits = message_logits + gumbel_sample
        gumbel_logits = message_logits.repeat(batch_size, 1, 1) + gumbel_sample
        '''
        u = torch.rand_like(x)*(0.999-self.eps)+self.eps
        gumbels = -torch.log( -torch.log(u))

        gumbel_logits = (x + gumbels) #/ (tau+eps)  # ~Gumbel(logits,tau)
        # Check shape:
        #print(f"Gumbel logits shape: {gumbel_logits.shape}")

        # Softmax with temperature
        # (batch_size x seq_len x vocab_dim )
        y_soft = F.softmax(gumbel_logits / eff_temperature, dim=-1)

        # Straight-through: use hard in forward, soft in backward
        if self.stgs_hard:
          #indices = torch.argmax(y_soft, dim=-1)
          # Sampling from batched distribution y_soft:
          message_ids = torch.distributions.Categorical(probs=y_soft).sample()
          y_hard = F.one_hot(message_ids, num_classes=self.vocab_size)
          # Type: half or full
          y_hard = y_hard.half() if x.dtype == torch.half else y_hard.float()
          # Straight-through trick: y_hard - y_soft.detach() + y_soft
          message_one_hot = y_hard - y_soft.detach() + y_soft
        else:
          message_ids = torch.distributions.Categorical(probs=y_soft).sample()
          message_one_hot = y_soft
        
        # Type: half or full
        message_one_hot = message_one_hot.half() if x.dtype == torch.half else message_one_hot.float()

        return message_ids, message_one_hot, eff_temperature


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

    stgs = STGS(
        vocab_size=allowed_vocab_size,
        stgs_hard=stgs_hard,
        init_temperature=temperature,
        learnable_temperature=learnable_temperature,
        eps=eps,
        device=device,
    )
    parameters += list(stgs.parameters())
    # check parameters contain stgs 

    if bptt:
        bptt_stgs = STGS(
            vocab_size=allowed_vocab_size,
            stgs_hard=bptt_stgs_hard,
            init_temperature=bptt_temperature,
            learnable_temperature=bptt_learnable_temperature,
            eps=bptt_eps,
            conditioning_dim=hidden_state_dim if bptt_hidden_state_conditioning else 0,
            device=device,
        )
        parameters += list(bptt_stgs.parameters())

    # Set up optimizer
    optimizer = optim.Adam(parameters, lr=learning_rate)

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

    # Training loop
    losses = []
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        # Apply ST-GS:
        message_logits = learnable_inputs.repeat(batch_size, 1, 1)
        # Type: half or full
        message_logits = message_logits.half() if kwargs['model_precision'] == "half" else message_logits
        message_ids, message_one_hot, eff_temperature  = stgs.forward(message_logits)
        
        prompt_ids = message_ids

        # Convert one-hot-like vectors to embeddings by manual matrix multiplication
        # learnable_inputs shape: [batch_size, seq_len, vocab_size]
        # embedding_weights shape: [vocab_size, embedding_dim]
        # Result shape: [batch_size, seq_len, embedding_dim]
        #input_embeddings = torch.matmul(learnable_inputs, embedding_weights)
        #input_embeddings = torch.matmul(message_one_hot, embedding_weights)
        # Convert learnable one-hot-like vectors to embeddings using the allowed subset
        # WARNING: TODO : PREVIOUSLY:
        #input_embeddings = torch.matmul(learnable_inputs, embedding_weights_subset)  # (batch, seq_len, embed_dim)
        input_embeddings = torch.matmul(message_one_hot, embedding_weights_subset)  # (batch, seq_len, embed_dim)

        # Get the model's embedding layer
        embedding_layer = model.get_input_embeddings()

        # Process pre-prompt if provided and combine with learnable embeddings
        if pre_prompt is not None:
            # Tokenize the pre-prompt
            pre_prompt_tokens = tokenizer(pre_prompt, return_tensors="pt").input_ids.to(device)

            # Convert tokens to embeddings
            pre_prompt_embeds = embedding_layer(pre_prompt_tokens).repeat(batch_size, 1, 1)

            # Combine pre-prompt embeddings with learnable embeddings
            # [batch_size, pre_prompt_length + learnable_length, hidden_size]
            combined_embeds = torch.cat([pre_prompt_embeds, input_embeddings], dim=1)
            current_embeds = combined_embeds
        else:
            # Use just the learnable embeddings
            current_embeds = input_embeddings

        # Generate completions from the model
        # Forward pass through the model with our embeddings
        outputs = model(
            inputs_embeds=current_embeds,
            output_hidden_states=True,
            use_cache=True,
            return_dict=True,
        )
        
        # Get the logits from the output
        logits = outputs.logits
        # Get the hidden states from the output
        hidden_states = outputs.hidden_states
        # Check shape:
        #print(f"Logits shape: {logits.shape}")
        # Restrict logits to allowed vocabulary
        logits_allowed = logits[..., allowed_tokens]  # (batch, seq_len, allowed_vocab_size)

        prompt_logits = logits_allowed
        #(batch_size x prompt_seq_len x vocab_size)

        # Initialize the past key values for generation
        past_key_values = outputs.past_key_values

        # Store all generated token logits
        #all_logits = [logits[:, -1:, :]]  # Start with the last logit from initial forward pass
        all_logits = [logits_allowed[:, -1:, :]]
        all_hidden_states = [hidden_states]

        # Generate additional tokens autoregressively to match target length
        current_length = 1  # We've already generated one token worth of logits
        
        completion_ids = []
        while current_length < target_length:
            # Get the predicted token ID from the last position
            if bptt:
                next_token_id, next_token_one_hot, bptt_eff_temperature = bptt_stgs(all_logits[-1], hidden_states=outputs.hidden_states)
                next_token_embedding = torch.matmul(next_token_one_hot, embedding_weights_subset)  # (batch, seq_len, embed_dim)
            else:
                bptt_eff_temperature = 0
                next_token_id = torch.argmax(all_logits[-1], dim=-1)
                # Get the embedding for this token
                if filter_vocab:
                    #checked that these are equivalent when not doing filtering...
                    next_token_embedding = embedding_weights_subset[next_token_id]
                else:
                    next_token_embedding = embedding_layer(next_token_id)
            #assert (next_token_embedding == next_token_embedding_2).all()
            completion_ids.append(next_token_id)

            # Forward pass with past key values for efficient generation
            outputs = model(
                inputs_embeds=next_token_embedding,
                past_key_values=past_key_values,
                output_hidden_states=True,
                use_cache=True,
                return_dict=True,
            )

            # Update past key values for next iteration
            past_key_values = outputs.past_key_values

            # Add the new logits to our collection
            next_logits = outputs.logits[..., allowed_tokens]
            all_logits.append(next_logits)
            all_hidden_states.append(outputs.hidden_states)

            current_length += 1

        # Concatenate all logits
        generated_logits = torch.cat(all_logits, dim=1)
        # Check shape:
        #print(f"Generated logits shape: {generated_logits.shape}")
        #print(f"Target tokens shape: {target_tokens_mapped.shape}")
        completion_ids = torch.cat(completion_ids, dim=1)
        # (batch_size x target_seq_len)
        # Compute loss against target tokens
        # We want to compare the generated token logits against the target tokens
        '''
        loss = loss_fn(
            #generated_logits.reshape(-1, vocab_size),
            generated_logits.reshape(-1, allowed_vocab_size),
            target_tokens_mapped.reshape(-1), #.reshape(1, -1).repeat(batch_size, 1),
            #target_tokens.reshape(-1)
        )
        '''
        losses_dict = loss_instance.compute_loss(
            input_dict={
                'generated_logits':generated_logits,#.reshape(-1,allowed_vocab_size),
                'generated_hidden_states': all_hidden_states,
                'completion_ids': completion_ids,
                'prompt_ids': prompt_ids,
                'prompt_logits': prompt_logits,
            },
        )
        loss = losses_dict['sumloss']

        # Check shape: expect none because reduction=mean is default
        #print(f"Loss shape: {loss.shape}")
        # Backward pass and optimize
        loss.backward()

        #Gradient clipping:
        if max_gradient_norm != 0.0:
           torch.nn.utils.clip_grad_norm_(parameters, max_gradient_norm)


        # Check gradient:
        #print()"Gradient shape: {learnable_inputs.grad.shape}")
        info = f"Gradient norm: {learnable_inputs.grad.norm().item():.6f} / Tau = {eff_temperature.item():.6f} / Allowed vocab size: {allowed_vocab_size}"

        # Check if any gradients are non-zero
        non_zero_grads = (learnable_inputs.grad != 0).sum().item()
        #print(f"Number of non-zero gradients: {non_zero_grads}")

        if non_zero_grads > 0:
            # Show some stats about the gradient distribution
            grad_abs = learnable_inputs.grad.abs()
            #print(f"Mean absolute gradient: {grad_abs.mean().item():.8f}")
            #print(f"Max absolute gradient: {grad_abs.max().item():.8f}")
            #print(f"Min absolute gradient (non-zero): {grad_abs[grad_abs > 0].min().item() if (grad_abs > 0).any() else 0:.8f}")

        # Optimisation
        optimizer.step()

        # Update progress bar
        losses.append(loss.item())
        pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f} / {info}")
        # Log metrics to wandb
        wandb_log = {
            "epoch": epoch+1,
            "loss": loss.item(),
            "effective_temperature": eff_temperature.item(),
            "bptt_effective_temperature": bptt_eff_temperature.mean().item() if isinstance(bptt_eff_temperature, torch.Tensor) else bptt_eff_temperature,
            "allowed_vocab_size": allowed_vocab_size,
            "non_zero_grads": non_zero_grads,
            "grad_mean": learnable_inputs.grad.mean().item() if learnable_inputs.grad is not None else 0.0,
            "grad_max": learnable_inputs.grad.max().item() if learnable_inputs.grad is not None else 0.0,
            "grad_min": (grad_abs[grad_abs > 0].min().item() if (grad_abs > 0).any() else 0.0),
            "grad_norm": learnable_inputs.grad.norm().item() if learnable_inputs.grad is not None else 0.0,
            "grad_std": learnable_inputs.grad.std().item() if learnable_inputs.grad is not None else 0.0,
            "vocab_size": model.config.vocab_size,

        }
        for k, v in losses_dict.items():
            wandb_log[k] = v.item()

        # Update wandb_table with generated_output:
        learnable_input_ids = torch.argmax(learnable_inputs, dim=-1)[0]
        generated_output_ids = torch.argmax(generated_logits, dim=-1)
        # Check shape:
        #print(f"Generated output ids shape: {generated_output_ids.shape}")
        # Remapping from allowed ids to original ids:
        table_generated_output_ids = generated_output_ids[0:1]
        table_generated_output_ids = torch.gather(allowed_tokens.unsqueeze(0), dim=1, index=table_generated_output_ids)
        learnable_input_str = tokenizer.decode(learnable_input_ids, skip_special_tokens=False)
        generated_output_str = tokenizer.decode(table_generated_output_ids[0], skip_special_tokens=False)
        #print(learnable_input_str)
        
        generated_tokens = table_generated_output_ids[0].cpu().tolist()

        metrics_dict = token_overlap_metric.measure(
            prompt_tokens=learnable_input_ids,
        )
        for k,v in metrics_dict.items():
            wandb_log[k] = v

        wandb_table.add_data(
            epoch+1, 
            target_text,
            learnable_input_ids.tolist(),
            learnable_input_str,
            table_generated_output_ids[0].tolist(), 
            generated_output_str,
            #token_overlap_measure,
            *metrics_dict.values(),
        )
        
        #TODO
        #for k,v in wandb_log.items():
        #    print(k, type(v))
        #import ipdb; ipdb.set_trace()
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

