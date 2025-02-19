"""This module contains the functions to read and write files."""

import time

import yaml


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.
    Args:
        config_path (str): The path to the YAML configuration file.
    Returns:
        dict: The contents of the configuration file as a dictionary.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def timeit(func):
    """
    A decorator that measures the execution time of a function.
    Args:
        func (callable): The function to be timed.
    Returns:
        callable: The wrapped function with timing functionality.
    The decorator prints the time taken to execute the function. If the time
    taken is less than 60 seconds, it prints the time in seconds. Otherwise,
    it prints the time in minutes.
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        elapsed_time = end - start
        if elapsed_time < 60:
            print(f"Time taken: {elapsed_time:.4f} seconds")
        else:
            print(f"Time taken: {elapsed_time/60:.4f} minutes")
        return result

    return wrapper
