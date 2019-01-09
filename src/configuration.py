import configparser
from os.path import dirname, realpath, join

# constants
CONFIG_FILE_PATH = 'config.ini'


def prepare_absolute_config_file_path():
    dir_of_file = dirname(realpath(__file__))

    return join(dir_of_file, CONFIG_FILE_PATH)


def open_config_file():
    config = configparser.ConfigParser()
    config.read(prepare_absolute_config_file_path())

    return config


def read_configuration(section, key):
    config = open_config_file()
    try:
        return config[section][key]
    except KeyError as e:
        print(e)
        return None


def update_configuration(section, key, value):
    config = open_config_file()
    config[section][key] = value

    with open(CONFIG_FILE_PATH, 'w') as configfile:
        config.write(configfile)


def read_database_path():
    return read_configuration('data', 'source_path')
