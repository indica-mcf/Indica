import configparser

def read_settings(file_path):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the .ini file
    config.read(file_path)

    # Initialize a dictionary to store the configurations
    config_dict = {}

    # Iterate over sections and items to populate the dictionary
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            config_dict[section][key] = value

    return config_dict

def write_settings(file_path, section, key, value):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the existing file
    config.read(file_path)

    # Check if the section exists, if not add it
    if section not in config.sections():
        config.add_section(section)

    # Add or update the key-value pair in the specified section
    config.set(section, key, value)

    # Write changes back to file
    with open(file_path, 'w') as configfile:
        config.write(configfile)

