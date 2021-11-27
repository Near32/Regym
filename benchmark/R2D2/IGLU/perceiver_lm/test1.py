import torch

from perceiver_io.decoders import PerceiverDecoder
from perceiver_io.encoder import PerceiverEncoder
from perceiver_io.perceiver_lm import PerceiverLM

import torch.nn as nn
import torch

num_latents = 128
latent_dim = 256
input_dim = 64

decoder_query_dim = 4

encoder = PerceiverEncoder(
    num_latents=num_latents,
    latent_dim=latent_dim,
    input_dim=input_dim,
    num_self_attn_per_block=8,
    num_blocks=1
)

decoder = PerceiverDecoder(
    latent_dim=latent_dim,
    query_dim=decoder_query_dim
)

#perceiver = PerceiverLM(encoder, decoder)
model = PerceiverLM(
    vocab_size=262, 
    max_seq_len=2048, 
    embedding_dim=768, 
    num_latents=256, 
    latent_dim=1280, 
    qk_out_dim=256, 
    num_self_attn_per_block=26,
)

#inputs = torch.randn(2, 16, input_dim)
#output_query = torch.randn(2, 3, decoder_query_dim)
#perceiver(inputs, output_query)  # shape = (2, 3, 4)
#print("done")


import pickle
with open("deepmind_assets/language_perceiver_io_bytes.pickle", "rb") as f:
    params = pickle.loads(f.read())


state_dict = {}
model_enc_base = 'perceiver.encoder.'
params_enc_base = 'perceiver_encoder/~/'

state_dict['token_embedding.weight'] = params['embed']['embeddings']
state_dict['decoder_token_bias'] = params['embedding_decoder']['bias']
state_dict['position_embedding.weight'] = params['trainable_position_encoding']['pos_embs']
state_dict['query_embedding.weight'] = params['basic_decoder/~/trainable_position_encoding']['pos_embs']
state_dict[f'{model_enc_base}latents'] = params[f'{params_enc_base}trainable_position_encoding']['pos_embs']

def copy_attention_params(model_base, params_base):
    global state_dict
    state_dict[f'{model_base}attention.q.weight'] = params[f'{params_base}attention/linear']['w'].T
    state_dict[f'{model_base}attention.q.bias'] = params[f'{params_base}attention/linear']['b']
    state_dict[f'{model_base}attention.k.weight'] = params[f'{params_base}attention/linear_1']['w'].T
    state_dict[f'{model_base}attention.k.bias'] = params[f'{params_base}attention/linear_1']['b']
    state_dict[f'{model_base}attention.v.weight'] = params[f'{params_base}attention/linear_2']['w'].T
    state_dict[f'{model_base}attention.v.bias'] = params[f'{params_base}attention/linear_2']['b']
    state_dict[f'{model_base}attention.projection.weight'] = params[f'{params_base}attention/linear_3']['w'].T
    state_dict[f'{model_base}attention.projection.bias'] = params[f'{params_base}attention/linear_3']['b']

    if 'self_attention' in params_base:
        state_dict[f'{model_base}layer_norm.weight'] = params[f'{params_base}layer_norm']['scale']
        state_dict[f'{model_base}layer_norm.bias'] = params[f'{params_base}layer_norm']['offset']
        state_dict[f'{model_base}qkv_layer_norm.weight'] = params[f'{params_base}layer_norm_1']['scale']
        state_dict[f'{model_base}qkv_layer_norm.bias'] = params[f'{params_base}layer_norm_1']['offset']
    else:
        state_dict[f'{model_base}q_layer_norm.weight'] = params[f'{params_base}layer_norm']['scale']
        state_dict[f'{model_base}q_layer_norm.bias'] = params[f'{params_base}layer_norm']['offset']
        state_dict[f'{model_base}kv_layer_norm.weight'] = params[f'{params_base}layer_norm_1']['scale']
        state_dict[f'{model_base}kv_layer_norm.bias'] = params[f'{params_base}layer_norm_1']['offset']
        state_dict[f'{model_base}qkv_layer_norm.weight'] = params[f'{params_base}layer_norm_2']['scale']
        state_dict[f'{model_base}qkv_layer_norm.bias'] = params[f'{params_base}layer_norm_2']['offset']

    state_dict[f'{model_base}mlp.mlp.0.weight'] = params[f'{params_base}mlp/linear']['w'].T
    state_dict[f'{model_base}mlp.mlp.0.bias'] = params[f'{params_base}mlp/linear']['b']
    state_dict[f'{model_base}mlp.mlp.2.weight'] = params[f'{params_base}mlp/linear_1']['w'].T
    state_dict[f'{model_base}mlp.mlp.2.bias'] = params[f'{params_base}mlp/linear_1']['b']

copy_attention_params(f'{model_enc_base}cross_attn.', f'{params_enc_base}cross_attention/')
copy_attention_params(f'perceiver.decoder.cross_attention.', f'basic_decoder/cross_attention/')

for i in range(26):
    copy_attention_params(f'{model_enc_base}self_attention_block.{i}.', f'{params_enc_base}self_attention{"_%d"%i if i else ""}/')
                                                                                                                                                                    
state_dict = {k: torch.tensor(v) for k,v in state_dict.items()}

print(model.load_state_dict(state_dict))

from deepmind_assets import bytes_tokenizer
import numpy as np
MAX_SEQ_LEN = 2048
# The tokenizer is just UTF-8 encoding (with an offset)
tokenizer = bytes_tokenizer.BytesTokenizer()
input_str = "This is an incomplete sentence where some words are missing."
input_tokens = tokenizer.to_int(input_str)

print(input_tokens)

inputs = input_tokens[None]
input_mask = np.ones_like(inputs)

def pad(
    max_sequence_length: int, 
    inputs, 
    input_mask=None,
    ):
    input_len = inputs.shape[1]
    assert input_len <= max_sequence_length
    pad_len = max_sequence_length - input_len
    padded_inputs = np.pad(
        inputs,
        pad_width=((0, 0), (0, pad_len)),
        constant_values=tokenizer.pad_token,
    )
    if input_mask is not None:
        padded_mask = np.pad(
            input_mask,
            pad_width=((0, 0), (0, pad_len)),
            constant_values=0,
        )
    else:
        return padded_inputs

    return padded_inputs, padded_mask

inputs = pad(MAX_SEQ_LEN, inputs)

model.eval()
output, logit = model.forward(torch.tensor(inputs))

print(output.shape)
