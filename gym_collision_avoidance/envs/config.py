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
    PLAY_MODE           = False # Enable to see the trained agent in action (for testing)
    EVALUATE_MODE       = False # Enable to see the trained agent in action (for testing)

    #########################################################################
    # COLLISION AVOIDANCE PARAMETERS
    NUM_TEST_CASES = 8
    PLOT_EPISODES = False # with matplotlib, plot after each episode
    DT             = 0.2 # seconds between simulation time steps
    REWARD_AT_GOAL = 1.0 # Number of agents trying to get from start -> goal positions
    REWARD_COLLISION = -0.25 # Number of agents trying to get from start -> goal positions
    REWARD_GETTING_CLOSE   = -0.1 # Number of agents trying to get from start -> goal positions
    COLLISION_DIST = 0.0 # meters between agents' boundaries for collision
    GETTING_CLOSE_RANGE = 0.2 # meters between agents' boundaries for collision
    REWARD_MIN     = -100 # Reward Clipping
    REWARD_MAX     = 100 # Reward Clipping
    TRAIN_WITH_REGRESSION = False # Start training with regression phase before RL
    LOAD_REGRESSION = True # Initialize training with regression network

    MAX_NUM_AGENTS = 2
    MULTI_AGENT_ARCHS = ['RNN','WEIGHT_SHARING','VANILLA']
    MULTI_AGENT_ARCH = 'VANILLA'
    # MULTI_AGENT_ARCH = 'WEIGHT_SHARING'
    # MULTI_AGENT_ARCH = 'RNN'

    # #########################################################################
    # # ALGORITHM PARAMETER
    # DISCOUNT                = 0.9 # Discount factor
    # TIME_MAX                = int(4/DT) # Tmax
    # MAX_QUEUE_SIZE          = 100 # Max size of the queue
    # PREDICTION_BATCH_SIZE   = 128
    # IMAGE_WIDTH             = 84 # Input of the DNN
    # IMAGE_HEIGHT            = 84
    # EPISODES                = 400000 # Total number of episodes and annealing frequency
    # ANNEALING_EPISODE_COUNT = 400000

    # # OPTIMIZER PARAMETERS
    # OPT_RMSPROP, OPT_ADAM   = range(2) # Initialize optimizer types as enum
    # OPTIMIZER               = OPT_ADAM # Game choice: Either "game_grid" or "game_ale"
    # LEARNING_RATE_REGRESSION_START = 1e-4 # Learning rate
    # LEARNING_RATE_REGRESSION_END = 1e-4 # Learning rate
    # LEARNING_RATE_RL_START     = 1e-4 # Learning rate
    # LEARNING_RATE_RL_END     = 1e-4 # Learning rate
    # RMSPROP_DECAY           = 0.99
    # RMSPROP_MOMENTUM        = 0.0
    # RMSPROP_EPSILON         = 0.1
    # BETA_START              = 1e-4 # Entropy regularization hyper-parameter
    # BETA_END                = 1e-4
    # USE_GRAD_CLIP           = False # Gradient clipping
    # GRAD_CLIP_NORM          = 40.0
    # LOG_EPSILON             = 1e-6 # Epsilon (regularize policy lag in GA3C)
    # TRAINING_MIN_BATCH_SIZE = 100 # Training min batch size - increasing the batch size increases the stability of the algorithm, but make learning slower

    # #########################################################################
    # # LOG AND SAVE
    # TENSORBOARD                  = True # Enable TensorBoard
    # TENSORBOARD_UPDATE_FREQUENCY = 50 # Update TensorBoard every X training steps
    # SAVE_MODELS                  = True # Enable to save models every SAVE_FREQUENCY episodes
    # SAVE_FREQUENCY               = 1000 # Save every SAVE_FREQUENCY episodes
    # PRINT_STATS_FREQUENCY        = 1 # Print stats every PRINT_STATS_FREQUENCY episodes
    # STAT_ROLLING_MEAN_WINDOW     = 1000 # The window to average stats
    # RESULTS_FILENAME             = 'results.txt'# Results filename
    # NETWORK_NAME                 = 'network'# Network checkpoint name

    # #########################################################################
    # # MORE EXPERIMENTAL PARAMETERS 
    # MIN_POLICY = 0.0 # Minimum policy

    HOST_AGENT_OBSERVATION_LENGTH = 4 # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_LENGTH = 1 # num other agents
    AGENT_ID_LENGTH = 1 # id
    IS_ON_LENGTH = 1 # 0/1 binary flag

    HOST_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 1.0, 0.5]) # dist to goal, heading to goal, pref speed, radius
    HOST_AGENT_STD_VECTOR = np.array([5.0, 3.14, 1.0, 1.0]) # dist to goal, heading to goal, pref speed, radius
    OTHER_AGENT_AVG_VECTOR = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    OTHER_AGENT_STD_VECTOR = np.array([5.0, 5.0, 1.0, 1.0, 1.0, 5.0, 1.0]) # other px, other py, other vx, other vy, other radius, combined radius, distance between
    RNN_HELPER_AVG_VECTOR = np.array([0.0])
    RNN_HELPER_STD_VECTOR = np.array([1.0])
    IS_ON_AVG_VECTOR = np.array([0.0])
    IS_ON_STD_VECTOR = np.array([1.0])

    if MAX_NUM_AGENTS == 2:
        # NN input:
        # [dist to goal, heading to goal, pref speed, radius, other px, other py, other vx, other vy, other radius, combined radius, distance between]
        OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
        HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
        FULL_STATE_LENGTH = HOST_AGENT_OBSERVATION_LENGTH + (MAX_NUM_AGENTS - 1) * OTHER_AGENT_FULL_OBSERVATION_LENGTH
        FIRST_STATE_INDEX = 0
        MULTI_AGENT_ARCH = 'NONE'

        NN_INPUT_AVG_VECTOR = np.hstack([HOST_AGENT_AVG_VECTOR,OTHER_AGENT_AVG_VECTOR])
        NN_INPUT_STD_VECTOR = np.hstack([HOST_AGENT_STD_VECTOR,OTHER_AGENT_STD_VECTOR])


    # if MAX_NUM_AGENTS in [3,4]:
    if MAX_NUM_AGENTS > 2:
        if MULTI_AGENT_ARCH == 'RNN':
            # NN input:
            # [num other agents, dist to goal, heading to goal, pref speed, radius, 
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius]
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = RNN_HELPER_LENGTH + HOST_AGENT_OBSERVATION_LENGTH + (MAX_NUM_AGENTS - 1) * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 1

            NN_INPUT_AVG_VECTOR = np.hstack([RNN_HELPER_AVG_VECTOR,HOST_AGENT_AVG_VECTOR,np.repeat(OTHER_AGENT_AVG_VECTOR,MAX_NUM_AGENTS-1)])
            NN_INPUT_STD_VECTOR = np.hstack([RNN_HELPER_STD_VECTOR,HOST_AGENT_STD_VECTOR,np.repeat(OTHER_AGENT_STD_VECTOR,MAX_NUM_AGENTS-1)])

        elif MULTI_AGENT_ARCH in ['WEIGHT_SHARING','VANILLA']:
            # NN input:
            # [dist to goal, heading to goal, pref speed, radius, 
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius, is_on,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius, is_on,
            #   other px, other py, other vx, other vy, other radius, dist btwn, combined radius, is_on]
            OTHER_AGENT_FULL_OBSERVATION_LENGTH = OTHER_AGENT_OBSERVATION_LENGTH + IS_ON_LENGTH
            HOST_AGENT_STATE_SIZE = HOST_AGENT_OBSERVATION_LENGTH
            FULL_STATE_LENGTH = HOST_AGENT_OBSERVATION_LENGTH + (MAX_NUM_AGENTS - 1) * OTHER_AGENT_FULL_OBSERVATION_LENGTH
            FIRST_STATE_INDEX = 0
            
            NN_INPUT_AVG_VECTOR = np.hstack([HOST_AGENT_AVG_VECTOR,np.repeat(np.hstack([OTHER_AGENT_AVG_VECTOR,IS_ON_AVG_VECTOR]),MAX_NUM_AGENTS-1)])
            NN_INPUT_STD_VECTOR = np.hstack([HOST_AGENT_STD_VECTOR,np.repeat(np.hstack([OTHER_AGENT_STD_VECTOR,IS_ON_STD_VECTOR]),MAX_NUM_AGENTS-1)])
            
    FULL_LABELED_STATE_LENGTH = FULL_STATE_LENGTH + AGENT_ID_LENGTH
    NN_INPUT_SIZE = FULL_STATE_LENGTH