import Archi
import yaml 

from Archi.modules.utils import copy_hdict 

def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./rl_filmed_ther_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    assert 'RLHead' in model.modules.keys()
    assert 'InstructionGenerator' in model.modules.keys()
    assert isinstance(model.modules['InstructionGenerator'], Archi.modules.CaptionRNNModule)
    assert 'CommEncoder' in model.modules.keys()
    assert isinstance(model.modules['CommEncoder'], Archi.modules.EmbeddingRNNModule)
    assert 'FiLMedBlock1' in model.modules.keys()
    assert 'FiLMedBlock2' in model.modules.keys()
    assert 'CoreLSTM' in model.modules.keys()
    assert 'CoreLSTM' in model.stream_handler.placeholders['inputs'].keys()
    assert 'hidden' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'cell' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'iteration' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
   

def test_model_forward():
    try:
        config = yaml.safe_load(
            open("./rl_filmed_ther_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    import torch 

    batch_size = 2
    action_dim = config['hyperparameters']['action_dim'] 
    exp_length = config['hyperparameters']['temporal_dim']
    max_sentence_length = config['hyperparameters']['max_sentence_length']
    vocab_size = config['hyperparameters']['vocab_size']

    use_cuda = False

    inputs_dict = {
        'obs':torch.rand(batch_size, 3,64,64),
        'action': torch.randint(0, action_dim,size=(batch_size,1)),
        'rnn_states':{
            'legal_actions': [torch.rand(batch_size, action_dim)],
            **model.get_reset_states({"repeat":batch_size, "cuda":use_cuda}),
            "phi_body": {
                "extra_inputs": {
                    "dialog": [torch.randint(
                        low=0, 
                        high=vocab_size, 
                        size=(batch_size, max_sentence_length), #,1),
                    )],
                },
            },
            "critic_body": {
                "extra_inputs": {
                    "previous_reward": [torch.rand(batch_size,1)],
                    "previous_action": [torch.randint(0, action_dim, size=(batch_size, 1))],
                    "action_mask": [torch.randint(0, 1, size=(batch_size, action_dim))],
                },
            },
        },
    }

    prediction = model(**inputs_dict)
    
    print("Model's Predictions:")
    for k,v in prediction.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} : {v.shape}")
        elif isinstance(v, dict):
            for k1,v1 in v.items():
                print(f"{k}:{k1} : {type(v1)}")
        else:
            print(f"{k} : {type(v)}")

    output = model.output_stream_dict
    
    ## TESTING INSTRUCTION GENERATOR:
    inputs_dict1 = {
        'obs':torch.rand(batch_size, exp_length, 3,64,64),
        'action': torch.randint(0,action_dim,size=(batch_size,1)),
        'rnn_states':{
            'legal_actions': [torch.rand(batch_size,action_dim)],
            **model.get_reset_states({"repeat":batch_size, "cuda":use_cuda}),
            "phi_body": {
                "extra_inputs": {
                    "dialog": [torch.randint(
                        low=0, 
                        high=vocab_size, 
                        size=(batch_size, max_sentence_length), # ,1),
                    )],
                },
            },
            "gt_sentences": [torch.randint(
                low=0,
                high=vocab_size,
                size=(batch_size, max_sentence_length), #,1),
            )],
            "critic_body": {
                "extra_inputs": {
                    "previous_reward": [torch.rand(batch_size,1)],
                    "previous_action": [torch.randint(0, action_dim, size=(batch_size, 1))],
                    "action_mask": [torch.randint(0, 1, size=(batch_size, action_dim))],
                },
            },
        },   
    }
    
    prediction1 = model(
        **inputs_dict1, 
        pipelines={
            'instruction_generator': config['pipelines']['instruction_generator'],
        },
    )
    output1 = model.output_stream_dict

    print("Model's Predictions 1:")
    for k,v in prediction1.items():
        if isinstance(v, torch.Tensor):
            print(f"{k} : {v.shape}")
        elif isinstance(v, dict):
            for k1,v1 in v.items():
                print(f"{k}:{k1} : {type(v1)}")
        else:
            print(f"{k} : {type(v)}")

    assert 'output' in output['modules']['CoreLSTM']
    assert 'qa' in output['modules']['RLHead']
    assert 'ent' in output['modules']['RLHead']
    assert 'log_a' in output['modules']['RLHead']
    assert 'processed_input' in output['inputs']['ObsEncoder']
    assert output1['inputs']['InstructionGenerator']['processed_input0'][0].max() != 0.0    
    assert len(dict(model.named_parameters())) != 0
    
    print("Model's Parameters:")
    for np, p in model.named_parameters():
        print(np)

if __name__ == '__main__':
    test_model_loading()
    test_model_forward()

