import numpy as np

class Config:
    #########################################################################
    # GENERAL PARAMETERS
    COLLISION_AVOIDANCE = True
    continuous, discrete = range(2) # Initialize game types as enum
    ACTION_SPACE_TYPE   = continuous
    NET_ARCH            = 'NetworkVP_cnn' # Neural net architecture
    ALL_ARCHS           = ['NetworkVP_cnn','NetworkVP_cnn_2'] # Can add more model types here
    TRAIN_MODELS        = True # Enable to train
    LOAD_CHECKPOINT     = False # Load old models. Throws if the model doesn't exist
    NORMALIZE_INPUT     = True
    USE_DROPOUT         = False
    WEIGHT_SHARING      = False
    USE_REGULARIZATION  = False
    LOAD_FROM_BACKUP_DIR= False
    LOAD_EPISODE        = 0 # If 0, the latest checkpoint is loaded

    ANIMATE_EPISODES    = False
    TRAIN_MODE           = False # Enable to see the trained agent in action (for testing)
    PLAY_MODE           = False # Enable to see the trained agent in action (for testing)
    EVALUATE_MODE       = True # Enable to see the trained agent in action (for testing)
    TRAIN_ON_MULTIPLE_AGENTS = True
    # TRAIN_ON_MULTIPLE_AGENTS = False

    LASERSCAN_LENGTH = 512 # num range readings in one scan

    MAX_NUM_AGENTS_IN_ENVIRONMENT = 10
    MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
    # STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agent_states', 'use_ppo'] # 2-agent net
    STATES_IN_OBS = ['dist_to_goal', 'radius', 'heading_ego_frame', 'pref_speed', 'other_agents_states', 'use_ppo', 'num_other_agents', 'laserscan'] # LSTM
    STATES_NOT_USED_IN_POLICY = ['use_ppo', 'num_other_agents', 'other_agents_states']
    STATE_INFO_DICT = {'dist_to_goal': {'dtype': np.float32, 'size': 1, 'bounds': [-np.inf, np.inf], 'attr': 'get_agent_data("dist_to_goal")', 'std': np.array([5.], dtype=np.float32), 'mean': np.array([0.], dtype=np.float32)},
                        'radius': {'dtype': np.float32, 'size': 1, 'bounds': [0, np.inf], 'attr': 'get_agent_data("radius")', 'std': np.array([1.0], dtype=np.float32), 'mean': np.array([0.5], dtype=np.float32)},
                        'heading_ego_frame': {'dtype': np.float32, 'size': 1, 'bounds': [-np.pi, np.pi], 'attr': 'get_agent_data("heading_ego_frame")', 'std': np.array([3.14], dtype=np.float32), 'mean': np.array([0.], dtype=np.float32)},
                        'pref_speed': {
                            'dtype': np.float32,
                            'size': 1,
                            'bounds': [0, np.inf],
                            'attr': 'get_agent_data("pref_speed")',
                            'std': np.array([1.0], dtype=np.float32),
                            'mean': np.array([1.0], dtype=np.float32)
                            },
                        'num_other_agents': {
                            'dtype': np.float32,
                            'size': 1,
                            'bounds': [0, np.inf],
                            'attr': 'get_agent_data("num_other_agents_observed")',
                            'std': np.array([1.0], dtype=np.float32),
                            'mean': np.array([1.0], dtype=np.float32)
                            },
                        'other_agent_states': {
                            'dtype': np.float32,
                            'size': 7,
                            'bounds': [-np.inf, np.inf],
                            'attr': 'get_agent_data("other_agent_states")',
                            'std': np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32),
                            'mean': np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32)
                            },
                        'other_agents_states': {
                            'dtype': np.float32,
                            'size': (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT,7),
                            'bounds': [-np.inf, np.inf],
                            'attr': 'get_agent_data("other_agents_states")',
                            'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0], dtype=np.float32), (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT, 1)),
                            'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0], dtype=np.float32), (MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT, 1)),
                            },
                        'laserscan': {'dtype': np.float32, 'size': LASERSCAN_LENGTH, 'bounds': [0., 6.], 'attr': 'get_sensor_data("laserscan")', 'std': 5.*np.ones((LASERSCAN_LENGTH), dtype=np.float32), 'mean': 5.*np.ones((LASERSCAN_LENGTH), dtype=np.float32)},
                        'use_ppo': {
                            'dtype': np.float32,
                            'size': 1,
                            'bounds': [0., 1.],
                            'attr': 'get_agent_data_equiv("policy.str", "learning")'
                            }
                        }
    MEAN_OBS = {}; STD_OBS = {}
    for state in STATES_IN_OBS:
        if 'mean' in STATE_INFO_DICT[state]:
            MEAN_OBS[state] = STATE_INFO_DICT[state]['mean']
        if 'std' in STATE_INFO_DICT[state]:
            STD_OBS[state] = STATE_INFO_DICT[state]['std']

    LSTM_HIDDEN_SIZE = 16
    NUM_LAYERS = 2
    NUM_HIDDEN_UNITS = 64
    NETWORK = "mfe_network"
    GAMMA = 0.99
    LEARNING_RATE = 1e-3

    PLOT_CIRCLES_ALONG_TRAJ = False

    #########################################################################
    # COLLISION AVOIDANCE PARAMETER
    # USE_LASERSCAN_IN_OBSERVATION = True
    USE_LASERSCAN_IN_OBSERVATION = False
    NUM_TEST_CASES = 50
    PLOT_EPISODES = False # with matplotlib, plot after each episode
    PLOT_EVERY_N_EPISODES = 100 # for tensorboard visualization
    DT             = 0.2 # seconds between simulation time steps
    REWARD_AT_GOAL = 1.0 # reward given when agent reaches goal position
    REWARD_COLLISION_WITH_AGENT = -0.25 # reward given when agent collides with another agent
    REWARD_COLLISION_WITH_WALL = -0.25 # reward given when agent collides with wall
    REWARD_GETTING_CLOSE   = -0.1 # reward when agent gets close to another agent (unused?)
    REWARD_ENTERED_NORM_ZONE   = -0.05 # reward when agent enters another agent's social zone
    REWARD_TIME_STEP   = 0.0 # default reward given if none of the others apply (encourage speed)
    NUM_STEPS_IN_OBS_HISTORY = 1 # number of time steps to store in observation vector
    NUM_PAST_ACTIONS_IN_STATE = 0
    COLLISION_DIST = 0.0 # meters between agents' boundaries for collision
    GETTING_CLOSE_RANGE = 0.2 # meters between agents' boundaries for collision
    # TRAIN_WITH_REGRESSION = False # Start training with regression phase before RL
    # LOAD_REGRESSION = False # Initialize training with regression network
    MULTI_AGENT_ARCHS = ['RNN','WEIGHT_SHARING','VANILLA']
    # MULTI_AGENT_ARCH = 'VANILLA'
    # MULTI_AGENT_ARCH = 'WEIGHT_SHARING'
    MULTI_AGENT_ARCH = 'RNN'

    SENSING_HORIZON  = np.inf
    # SENSING_HORIZON  = 3.0

    HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1 # num other agents
    AGENT_ID_LENGTH = 1 # id, 0/1 binary flag of which policy it's using
    IS_ON_LENGTH = 1 # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5]) # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0]) # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    # if MAX_NUM_AGENTS_IN_ENVIRONMENT == 2:
    #     # NN input:
    #     # [dist to goal, heading to goal, pref speed, radius, other px, other py, other vx, other vy, other radius, combined radius, distance between]
    #     MAX_NUM_OTHER_AGENTS_OBSERVED = 1
    #     OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
    #     HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
    #     FULL_STATE_LENGTH = HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
    #     FIRST_STATE_INDEX = 0
    #     MULTI_AGENT_ARCH = 'NONE'

    #     NN_INPUT_AVG_VECTOR = np.hstack([HOST_AGENT_AVG_VECTOR, OTHER_AGENT_AVG_VECTOR])
    #     NN_INPUT_STD_VECTOR = np.hstack([HOST_AGENT_STD_VECTOR, OTHER_AGENT_STD_VECTOR])

    #     if USE_LASERSCAN_IN_OBSERVATION:
    #         FULL_STATE_LENGTH += LASERSCAN_LENGTH
    #         NN_INPUT_AVG_VECTOR = np.hstack([NN_INPUT_AVG_VECTOR, 4*np.ones(LASERSCAN_LENGTH)])
    #         NN_INPUT_STD_VECTOR = np.hstack([NN_INPUT_STD_VECTOR, 2*np.ones(LASERSCAN_LENGTH)])

    # if MAX_NUM_AGENTS in [3,4]:
    if MAX_NUM_AGENTS_IN_ENVIRONMENT >= 2:
        if MULTI_AGENT_ARCH == 'RNN':
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius, 
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            MAX_NUM_OTHER_AGENTS_OBSERVED = 10
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 1

            NN_INPUT_AVG_VECTOR = np.hstack([RNN_HELPER_AVG_VECTOR,HOST_AGENT_AVG_VECTOR,np.tile(OTHER_AGENT_AVG_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])
            NN_INPUT_STD_VECTOR = np.hstack([RNN_HELPER_STD_VECTOR,HOST_AGENT_STD_VECTOR,np.tile(OTHER_AGENT_STD_VECTOR,MAX_NUM_OTHER_AGENTS_OBSERVED)])

        elif MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
            # NN input:
            # [dist to goal, heading to goal, pref speed, radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius, is_on,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius, is_on,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius, is_on]
            MAX_NUM_OTHER_AGENTS_OBSERVED = MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH + IS_ON_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 0

            NN_INPUT_AVG_VECTOR = np.hstack([HOST_AGENT_AVG_VECTOR,np.tile(np.hstack([OTHER_AGENT_AVG_VECTOR,IS_ON_AVG_VECTOR]),MAX_NUM_OTHER_AGENTS_OBSERVED)])
            NN_INPUT_STD_VECTOR = np.hstack([HOST_AGENT_STD_VECTOR,np.tile(np.hstack([OTHER_AGENT_STD_VECTOR,IS_ON_STD_VECTOR]),MAX_NUM_OTHER_AGENTS_OBSERVED)])

    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH
    NN_INPUT_SIZE = FULL_STATE_LENGTH



    #     FULL_STATE_DIST_TO_GOAL_INDEX = 0
    #     FULL_STATE_HEADING_TO_GOAL_INDEX = 1
    #     FULL_STATE_PREF_SPEED_INDEX = 2
    #     FULL_STATE_RADIUS_INDEX = 3
    #     FULL_STATE_OTHER_PX_INDEX = 4
    #     FULL_STATE_OTHER_PY_INDEX = 5
    #     FULL_STATE_OTHER_VX_INDEX = 6
    #     FULL_STATE_OTHER_VY_INDEX = 7
    #     FULL_STATE_OTHER_RADIUS_INDEX = 8
    #     FULL_STATE_DIST_BETWEEN_INDEX = 9
    #     FULL_STATE_COMBINED_RADIUS_INDEX = 10
