from configparser import ConfigParser


def get_api_key():
    config = ConfigParser()
    config.read('api_key.config')
    return config['anthropic']['api_key']
