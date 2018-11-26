import configparser


# constants
CONFIG_FILE_PATH = 'config.ini'


def open_config_file():
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE_PATH)

    return config


def read_configuration(sections, key):
    config = open_config_file()

    try:
        return config[sections][key]
    except KeyError:
        return None


def update_configuration(section, key, value):
    config = open_config_file()
    config[section][key] = value

    with open(CONFIG_FILE_PATH, 'w') as configfile:
        config.write(configfile)


def read_database_path():
    return read_configuration('data', 'source_path')
