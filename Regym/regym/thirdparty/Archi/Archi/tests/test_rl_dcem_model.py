import Archi
import yaml 

from Archi.modules.utils import copy_hdict 

def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./rl_dcem_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    assert 'RLHead' in model.modules.keys()
    assert 'ObsMemory' in model.modules.keys()
    assert 'CommMemory' in model.modules.keys()
    assert 'memory' in model.stream_handler.placeholders['inputs']['ObsMemory'].keys()
    assert 'memory' in model.stream_handler.placeholders['inputs']['CommMemory'].keys()
    assert '0_read_value' in model.stream_handler.placeholders['inputs']['CommKObsVReadHeadsModule'].keys()
    assert '1_read_value' in model.stream_handler.placeholders['inputs']['CommKObsVReadHeadsModule'].keys()
    assert '2_read_value' in model.stream_handler.placeholders['inputs']['CommKObsVReadHeadsModule'].keys()
    assert 'CoreLSTM' in model.modules.keys()
    assert 'CoreLSTM' in model.stream_handler.placeholders['inputs'].keys()
    assert 'hidden' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'cell' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'iteration' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
   

def test_model_forward():
    try:
        config = yaml.safe_load(
            open("./rl_dcem_model_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    import torch 
    
    batch_size = 4
    use_cuda = False

    inputs_dict = {
        'obs':torch.rand(batch_size,3,64,64),
        'action': torch.randint(0,8,size=(4,1)),
        'rnn_states':{
            'comm':torch.rand(batch_size,15),
            'legal_actions': [torch.rand(4,8)],
            **model.get_reset_states({"repeat":batch_size, "cuda":use_cuda}),
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
    assert output['inputs']['CommKObsVReadHeadsModule']['0_read_value'][0].max() != 0.0    

    inputs_dict1 = {
        'obs':torch.rand(batch_size,3,64,64),
        'action': torch.randint(0,8,size=(4,1)),
        'rnn_states':{
            'comm':torch.rand(batch_size,15),
            'legal_actions': [torch.rand(4,8)],
            **output['inputs'],
        },
    }

    prediction1 = model(**inputs_dict1)
    output1 = model.output_stream_dict

    assert 'output' in output['inputs']['CoreLSTM']
    assert 'processed_input' in output['inputs']['ToOutputFCN']
    assert 'qa' in output['modules']['RLHead']
    assert 'ent' in output['modules']['RLHead']
    assert 'log_a' in output['modules']['RLHead']
    assert 'processed_input' in output['inputs']['ObsEncoder']
    assert 'processed_input' in output['inputs']['CommToCommQueryFCN']
    assert output['inputs']['CommKObsVReadHeadsModule']['0_read_value'][0].max() != output1['inputs']['CommKObsVReadHeadsModule']['0_read_value'][0].max()    
    assert len(dict(model.named_parameters())) != 0
    
    print("Model's Parameters:")
    for np, p in model.named_parameters():
        print(np)

if __name__ == '__main__':
    test_model_loading()
    test_model_forward()

