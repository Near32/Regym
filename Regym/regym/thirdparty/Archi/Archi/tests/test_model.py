import Archi
import yaml 


def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./model_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    assert 'FCNModule_0' in model.modules.keys()
    assert 'test_entry' in model.modules['FCNModule_0'].config.keys()
    assert 'CoreLSTM' in model.modules.keys()
    assert 'CoreLSTM' in model.stream_handler.placeholders['inputs'].keys()
    assert 'hidden' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'cell' in model.stream_handler.placeholders['inputs']['CoreLSTM'].keys()
    assert 'SecondaryDNC' in model.modules.keys()
    assert 'SecondaryDNC' in model.stream_handler.placeholders['inputs'].keys()
    assert 'dnc' in model.stream_handler.placeholders['inputs']['SecondaryDNC'].keys()
    assert 'dnc_body' in model.stream_handler.placeholders['inputs']['SecondaryDNC']['dnc'].keys()
    assert 'dnc_controller' in model.stream_handler.placeholders['inputs']['SecondaryDNC']['dnc'].keys()
    assert 'dnc_memory' in model.stream_handler.placeholders['inputs']['SecondaryDNC']['dnc'].keys()
    
def test_model_forward():
    try:
        config = yaml.safe_load(
            open("./model_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    import torch 

    batch_size = 4
    use_cuda = True 

    inputs_dict = {
        'obs': torch.rand(batch_size,3,64,64),
        'rnn_states': {
            'y':torch.rand(4,32),
            **model.get_reset_states({"repeat":batch_size, "cuda":use_cuda}),
        },
    }

    prediction = model(**inputs_dict)
    output = model.output_stream_dict

    inputs_dict1 = {
        'obs': torch.rand(batch_size,3,64,64),
        'rnn_states':{
            'y': [torch.rand(4,32)],
            **output['inputs'],
        },
    }

    prediction1 = model(**inputs_dict1)
    output1 = model.output_stream_dict

    assert 'processed_input' in output['inputs']['ConvNetModule_0']
    assert 'processed_input' in output['inputs']['FCNModule_0']
    assert 'output' in output['inputs']['CoreLSTM']
    assert 'dnc_output' in output['inputs']['SecondaryDNC']
    assert output1['inputs']['SecondaryDNC']['dnc']['dnc_body']['prev_read_vec'][0].max() != 0.0    
    assert len(dict(model.named_parameters())) != 0
    for np, p in model.named_parameters():
        print(np)

if __name__ == '__main__':
    test_model_loading()
    test_model_forward()

