# # import the necessary packages
# import os
# # initialize the path to the *original* input directory of images
# ORIG_INPUT_DATASET = "train"
# # initialize the base path to the *new* directory that will contain
# # our images after computing the training and testing split
# BASE_PATH = "disease_no_disease"
# # derive the training, validation, and testing directories
# TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
# VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
# TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])

# # define the amount of data that will be used training
# TRAIN_SPLIT = 0.75
# # the amount of validation data will be a percentage of the
# # *training* data
# VAL_SPLIT = 0.1
# # define the names of the classes
# CLASSES = ["1", "2","3","4","5"]

# # initialize the initial learning rate, batch size, and number of
# # epochs to train for
# INIT_LR = 1e-4
# BS = 32
# NUM_EPOCHS = 20
# # define the path to the serialized output model after training
# MODEL_PATH = "disease_detector.model"


# Learning rate parameters
BASE_LR = 0.001
EPOCH_DECAY = 30 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.


# DATASET INFO
NUM_CLASSES = 5 # set the number of classes in your dataset
DATA_DIR = 'data/' # to run with the sample dataset, just set to 'hymenoptera_data'

# DATALOADER PROPERTIES
BATCH_SIZE = 32 # Set as high as possible. If you keep it too high, you'll get an out of memory error.


### GPU SETTINGS
CUDA_DEVICE = 0 # Enter device ID of your gpu if you want to run on gpu. Otherwise neglect.
GPU_MODE = 0 # set to 1 if want to run on gpu.


# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 0 #if you want to use tensorboard set this to 1.
TENSORBOARD_SERVER = "YOUR TENSORBOARD SERVER ADDRESS HERE" # If you set.
EXP_NAME = "fine_tuning_experiment" # if using tensorboard, enter name of experiment you want it to be displayed as.