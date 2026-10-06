CLIP_MODEL = "google/siglip-base-patch16-224"
# CLIP_MODEL = "google/siglip-so400m-patch14-384"
PROCESSED_WIT = "/home/processed_wit"
CACHE_DIR = "/home/cache"
MAX_LENGTH = 64

WIT_DATA = ["https://storage.googleapis.com/gresearch/wit/wit_v1.train.all-00000-of-00010.tsv.gz",]


EMBED_SIZE = 1152
ORTHO_SIZE = 256
HIDDEN_SIZE = 1152
ENCODER_LAYERS = 3
DECODER_LAYERS = 6

ACTION_MODEL = "/gayu-west/action_model_{}"
ACTION_TRAIN_EPOCHS = 1000
ACTION_BATCH_SIZE = 64
ACTION_SAVE_STEPS = 10_000
ACTION_SAVE_LIMIT = 2

ACTION_LR = 1e-4
ACTION_LOGDIR_ORTHO = "/gayu-west/logs_{}"
ACTION_LOG_STEPS = 100

CMU_PLAY = "/home/cmu_fusion/0.1.0"
# PROCESSED_CMU_PLAY = "/home/processed/cmu_play"
# FAISS_INDEX_CMU_PLAY = "/home/processed/cmu_play.faiss"
PROCESSED_CMU_PLAY = "/gayu-west/data"
FAISS_INDEX_CMU_PLAY = "/gayu-west/data/index.faiss"

RAGTAG_MODEL = "/home/ragtag_model"
RAGTAG_TRAIN_EPOCHS = 400
RAGTAG_BATCH_SIZE = 64
RAGTAG_SAVE_STEPS = 250
RAGTAG_SAVE_LIMIT = 2

RAGTAG_LR = 1e-3
RAGTAG_LOGDIR = "/home/ragtag_logs"
RAGTAG_LOG_STEPS = 25


DEMONSTRATIONS_DIR = "/gayu-west/data"
DEMONSTRATIONS_DIR_VEL = "/gayu-west/data_vel_3_30"
DEMONSTRATIONS_DIR_EE = "/gayu-west/data_ee"
DEMONSTRATIONS_DIR_ACT = "/gayu-west/data_act_30"
DEMONSTRATIONS_DIR_VEL_4 = "/gayu-west/data_vel_3"
DEMONSTRATIONS_DIR_EE_DINO = "/gayu-west/data_ee_dino"
DEMONSTRATIONS_DIR_EE_PUSH_CUBE = "/gayu-west/data_ee_push_cube"
DEMONSTRATIONS_DIR_EE_PUSH_CUBE_DINO = "/gayu-west/data_ee_push_cube_dino"
DEMONSTRATIONS_DIR_EE_PUSH_CUBE_JEPA = "/gayu-west/data_ee_push_cube_jepa"
DEMONSTRATIONS_DIR_EE_PUSH_CUBE_JEPA_IMAGE = "/gayu-west/data_ee_push_cube_jepa_image"