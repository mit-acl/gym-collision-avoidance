# Find the config file (path provided as an environment variable),
# import and instantiate it here so all modules have access
import os
gym_config_path = os.environ.get('GYM_CONFIG_PATH', os.path.join(os.path.dirname(os.path.realpath(__file__)), 'config.py'))
gym_config_class = os.environ.get('GYM_CONFIG_CLASS', 'Config')

import sys
if sys.version[0] == '3':
	import importlib.util
	spec = importlib.util.spec_from_file_location(gym_config_class, gym_config_path)
	foo = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(foo)
elif sys.version[0] == '2':
	import imp
	foo = imp.load_source(gym_config_class.split('.')[-1], gym_config_path)
config_class = getattr(foo, gym_config_class, None)
assert(callable(config_class))
Config = config_class()
