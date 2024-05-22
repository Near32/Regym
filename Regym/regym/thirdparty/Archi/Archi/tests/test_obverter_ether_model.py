import Archi
import yaml 

import torch


def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./obverter_ether_test_config.yaml", 'r'),
        )
    except yaml.YAMLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    return

def test_model_speaker_forward():
    try:
        config = yaml.safe_load(
            open("./obverter_ether_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    import torch 

    batch_size = 4
    use_cuda = True 

    inputs_dict = {
        'obs': torch.rand(batch_size,64),
        'rnn_states': {
            **model.get_reset_states({"repeat":batch_size, "cuda":use_cuda}),
            'sentences_widx': None,
        },
    }

    prediction = model(**inputs_dict)
    output = model.output_stream_dict

    for k,v in prediction.items():
        print(k, v.shape if isinstance(v, torch.Tensor) else ' ... ')

    assert output['modules']['Obv_0']['sentences_widx'] is not None
    assert output['modules']['Obv_0']['sentences_logits'] is not None

    for np, p in model.named_parameters():
        print(np)

def test_model_listener_forward():
    try:
        config = yaml.safe_load(
            open("./obverter_ether_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    import torch 

    batch_size = 4
    use_cuda = True 

    inputs_dict = {
        'obs': torch.rand(batch_size,64),
        'rnn_states': {
            **model.get_reset_states({"repeat":batch_size, "cuda":use_cuda}),
            'sentences_widx': torch.randint(
                high=config['vocab_size'], 
                size=(batch_size, config['max_sentence_length'], 1),
            ),
        },
    }

    prediction = model(**inputs_dict)
    output = model.output_stream_dict

    for k,v in prediction.items():
        print(k, v.shape if isinstance(v, torch.Tensor) else ' ... ')

    assert output['modules']['Obv_0']['decision'] is not None

    for np, p in model.named_parameters():
        print(np)

if __name__ == '__main__':
    test_model_loading()
    test_model_speaker_forward()
    test_model_listener_forward()
