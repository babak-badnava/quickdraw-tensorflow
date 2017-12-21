class Config:

    #########################################################################

    TRAIN_MODELS = True
    # Load old models. Throws if the model doesn't exist
    LOAD_CHECKPOINT = True
    # If 0, the latest checkpoint is loaded
    LOAD_EPISODE = 99000

    #########################################################################
    # Algorithm parameters
    RUNNING_LOADER_COUNT = 1
    NB_TRAINER = 1

    # Max size of the queue
    MAX_QUEUE_SIZE = 100
    BATCH_SIZE = 64
    QUEUE_SIZE = 2000
    NB_SAMPLE_PER_CLASS = 5000

    # Input of the Network
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128
    NB_CLASSES = 3

    # Total number of episodes and annealing frequency
    EPOCH = 1000
    ANNEALING_EPOCH_COUNT = 400000

    # Learning rate
    LEARNING_RATE_START = 0.0003
    LEARNING_RATE_END = 0.0003

    #########################################################################
    # Log and save

    # Enable TensorBoard
    TENSORBOARD = False
    # Update TensorBoard every X training steps
    TENSORBOARD_UPDATE_FREQUENCY = 1000

    # Enable to save models every SAVE_FREQUENCY episodes
    SAVE_MODELS = True
    # Save every SAVE_FREQUENCY episodes
    SAVE_FREQUENCY = 1000

    # Network checkpoint name
    NETWORK_NAME = 'network'

    # Device
    DEVICE = 'gpu:0'
