# BAT SETTINGS
BAT_SPEED = 4
BAT_OBSERVABLE_DISTANCE = 25
ANGLE_RANGE = 20

# ENV SETTINGS
OFFSET = BAT_SPEED * 2
PRINT_PATH = True
NUM_OBSTABLES = 40
RANDOM_OBSTACLES = True  # Ture => Use random obstacles
MARGIN_NO_OBSTICLE = 20  # Form the boundaries where the obsticle exist
SITE_MARGIN = 1
LOAD_SAND = True  # False => No obstacles
OUTPUT_FILE = 'tmp.csv'
SEED = 101
SHAPE_FILE = 'shape'
GAME_SIZE = 401
# Determine where the goal's x coordinate
MARGIN_TO_GOAL_X_AXIS = BAT_SPEED * 10

# REWARD SETTINGS
REWARD_HIT_TREE = -1  # -1
REWARD_MOVE = -0.3
REWARD_BETTER_DISTANCE = 0.2
REWARD_ON_EDGE = -0.9
REWARD_GOAL = 10  # 5


# MODEL SETTINGS
MODEL_FILE = 'tmp_brain.pth'
GAMMA = 0.96  # discount rate
LEARNING_RATE = 0.0001
BATCH_SIZE = 200  # training batch size
TARGET_UPDATE = 100  # number of moves to update the target network
