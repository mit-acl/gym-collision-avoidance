import numpy as np

class Config:
    #########################################################################
    # GENERAL PARAMETERS
    continuous, discrete = range(2) # Initialize game types as enum
    ACTION_SPACE_TYPE   = continuous
    NET_ARCH            = 'NetworkVP_cnn' # Neural net architecture
    ALL_ARCHS           = ['NetworkVP_cnn','NetworkVP_cnn_2'] # Can add more model types here
    TRAIN_MODELS        = True # Enable to train
    LOAD_CHECKPOINT     = True # Load old models. Throws if the model doesn't exist
    NORMALIZE_INPUT     = True
    USE_DROPOUT         = False
    WEIGHT_SHARING      = True
    USE_REGULARIZATION  = True
    LOAD_FROM_BACKUP_DIR= False
    LOAD_EPISODE        = 0 # If 0, the latest checkpoint is loaded

    ANIMATE_EPISODES    = False
    PLOT_EPISODES       = False
    TRAIN_MODE           = False # Enable to see the trained agent in action (for testing)
    PLAY_MODE           = False # Enable to see the trained agent in action (for testing)
    EVALUATE_MODE       = False # Enable to see the trained agent in action (for testing)
    #  TRAIN_ON_MULTIPLE_AGENTS = True
    TRAIN_ON_MULTIPLE_AGENTS = False

    #########################################################################
    # COLLISION AVOIDANCE PARAMETER
    USE_ROS = True
    #  USE_ROS = True
    MAX_NUM_AGENTS_IN_ENVIRONMENT = 2
    NUM_TEST_CASES = 8
    PLOT_EPISODES = False # with matplotlib, plot after each episode
    PLOT_EVERY_N_EPISODES = 100 # for tensorboard visualization
    DT             = 0.2 # seconds between simulation time steps
    REWARD_AT_GOAL = 1.0 # Number of agents trying to get from start -> goal positions
    REWARD_COLLISION = -0.25 # Number of agents trying to get from start -> goal positions
    REWARD_GETTING_CLOSE   = -0.1 # Number of agents trying to get from start -> goal positions
    REWARD_NORM_VIOLATION   = -0.05 # Number of agents trying to get from start -> goal positions
    NUM_AGENT_STATES = 4 # Number of states (pos_x,pos_y,...)
    OTHER_OBS_LENGTH = 7 # number of states about another agent in observation vector
    NUM_STEPS_IN_OBS_HISTORY = 1 # number of time steps to store in observation vector
    NUM_PAST_ACTIONS_IN_STATE = 0
    COLLISION_DIST = 0.0 # meters between agents' boundaries for collision
    GETTING_CLOSE_RANGE = 0.2 # meters between agents' boundaries for collision
    STACKED_FRAMES = 1 # Num of inputs to DQN
    REWARD_MIN     = -100 # Reward Clipping
    REWARD_MAX     = 100 # Reward Clipping
    MAX_ITER       = 40 # Max iteration (time limit)
    TIMER_DURATION = 0.01 # In second visualization time for each step
    ACTIONS_FROM_VALUE_FUNCTION = False # In second visualization time for each step
    ACTIONS_FROM_POLICY_FUNCTION = True # In second visualization time for each step
    TRAIN_WITH_REGRESSION = False # Start training with regression phase before RL
    LOAD_REGRESSION = False # Initialize training with regression network
    MULTI_AGENT_ARCHS = ['RNN','WEIGHT_SHARING','VANILLA']
    MULTI_AGENT_ARCH = 'VANILLA'
    # MULTI_AGENT_ARCH = 'WEIGHT_SHARING'
    #  MULTI_AGENT_ARCH = 'RNN'

    SENSING_HORIZON  = np.inf
    # SENSING_HORIZON  = 3.0

    HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1 # num other agents
    AGENT_ID_LENGTH = 2 # id, 0/1 binary flag of which policy it's using
    IS_ON_LENGTH = 1 # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5]) # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0]) # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    if MAX_NUM_AGENTS_IN_ENVIRONMENT == 2:
        # NN input:
        # [dist to goal, heading to goal, pref speed, radius, other px, other py, other vx, other vy, other radius, combined radius, distance between]
        MAX_NUM_OTHER_AGENTS_OBSERVED = 1
        OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
        HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
        FULL_STATE_LENGTH = HOST_AGENT_OBSERVATION_LENGTH + MAX_NUM_OTHER_AGENTS_OBSERVED * OTHER_AGENT_FULL_OBSERVATION_LENGTH
        FIRST_STATE_INDEX = 0
        MULTI_AGENT_ARCH = 'NONE'

        NN_INPUT_AVG_VECTOR = np.hstack([HOST_AGENT_AVG_VECTOR,OTHER_AGENT_AVG_VECTOR])
        NN_INPUT_STD_VECTOR = np.hstack([HOST_AGENT_STD_VECTOR,OTHER_AGENT_STD_VECTOR])


    # if MAX_NUM_AGENTS in [3,4]:
    if MAX_NUM_AGENTS_IN_ENVIRONMENT > 2:
        if MULTI_AGENT_ARCH == 'RNN':
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius, 
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            MAX_NUM_OTHER_AGENTS_OBSERVED = 3
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
            MAX_NUM_OTHER_AGENTS_OBSERVED = 2
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
