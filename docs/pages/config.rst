.. _config:

Simulation Settings
========================================

All settings are controlled by :code:`config.py`. There is a default :code:`Config` class with settings such as, whether to plot episodes, simulation timestep, etc.

If you have your own experiment in mind, you should create a subclass of :code:`Config` to overwrite the default settings. For example, if you want to keep all defaults but also save a plot from each episode:

.. parsed-literal::
    from gym_collision_avoidance.envs.config import Config as EnvConfig
    class NewExperiment(EnvConfig):
        def __init__(self):
            EnvConfig.__init__(self)
            self.SAVE_EPISODE_PLOTS = True

.. warning::
    It's a little sketchy whether you should put your changes before/after the initialization of the superclass. Some attributes need to be set before the superclass is initialized because they are used in determining other settings, and others will be overwritten if you do this... it may be safest to put them before and after for now? Sorry about this.

In your experiment script, before creating an instance of the environment, set the environment variables that point to your new config:

    .. parsed-literal::
        import os
        os.environ['GYM_CONFIG_CLASS'] = 'NewExperiment'

        # If your new config class is not in config.py, set this:
        os.environ['GYM_CONFIG_PATH'] = 'path_to_file_containing_your_new_config_class'

You can see examples of experiment config classes in :code:`config.py`, such as :code:`Formations`

I am open to suggestions on better ways of handling the settings file.