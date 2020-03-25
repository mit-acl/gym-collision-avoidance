# Find the config file (path provided as an environment variable),
# import and instantiate it here so all modules have access
import os
gym_config_path = os.environ['GYM_CONFIG_PATH']
gym_config_class = os.environ['GYM_CONFIG_CLASS']

import importlib.util
spec = importlib.util.spec_from_file_location(gym_config_class, gym_config_path)
foo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foo)
config_class = getattr(foo, gym_config_class, None)
assert(callable(config_class))
Config = config_class()
