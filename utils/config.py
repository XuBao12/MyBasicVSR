import yaml

def load_config(config_file):
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # _merge(defaults, config)
    return config
