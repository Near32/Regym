import Archi
import yaml 


def test_model_loading():
    try:
        config = yaml.safe_load(
            open("./referentialgym_test_config.yaml", 'r'),
        )
    except yaml.YANNLError as e:
        print(e)

    from Archi import load_model

    model = load_model(config)
    
    return


