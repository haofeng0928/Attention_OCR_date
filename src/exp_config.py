import platform

"""
Default paramters for experiemnt
"""


class ExpConfig:

    PHASE = 'train'
    GPU_ID = 2
    CHANNEL = 1  # 3
    MEAN = [128]  # [179, 178, 179]
    VISUALIZE = False
    USE_GRU = True
    LOAD_MODEL = True
    OLD_MODEL_VERSION = False

    # input and output for training
    # if PHASE == 'train':
    #     DATA_DIR = r'D:\myData\huawei_datetext\train_img'
    #     LABEL_PATH = r'D:\myData\huawei_datetext\train_txt.txt'
    # elif PHASE == 'test':
    #     DATA_DIR = r'D:\myData\huawei_datetext\val_img'
    #     LABEL_PATH = r'D:\myData\huawei_datetext\val_txt.txt'
    if PHASE == 'train':
        DATA_DIR = '/home/f50004690/myData/huawei_datetext/train_img'
        LABEL_PATH = '/home/f50004690/myData/huawei_datetext/train_txt.txt'
    elif PHASE == 'test':
        DATA_DIR = '/home/f50004690/myData/huawei_datetext/val_img'
        LABEL_PATH = '/home/f50004690/myData/huawei_datetext/val_txt.txt'
    LEXICON = 'date_lexicon.txt'
    MODEL_DIR = 'models'  # the directory for saving and loading model parameters (structure is not stored)
    LOG_PATH = 'log.txt'
    OUTPUT_DIR = 'results'  # output directory
    STEPS_PER_CHECKPOINT = 100  # 500  # checkpointing (print perplexity, save model)

    # Optimization
    NUM_EPOCH = 1  # 1000
    BATCH_SIZE = 32
    INITIAL_LEARNING_RATE = 1.0

    # Network parameters
    CLIP_GRADIENTS = True  # whether to perform gradient clipping
    MAX_GRADIENT_NORM = 5.0  # Clip gradients to this norm
    ATTN_USE_LSTM = True  # 未使用 # whether or not use LSTM attention decoder cell
    TARGET_EMBEDDING_SIZE = 10  # embedding dimension for each target
    TARGET_VOCAB_SIZE = 16  # 26+10+3  # 0:PADDING, 1:GO, 2:EOS, >2:0-9,a-z

    # (Encoder number of hidden units will be ATTN_NUM_HIDDEN*ATTN_NUM_LAYERS)
    ATTN_NUM_HIDDEN = 256  # number of hidden units in attention decoder cell
    ATTN_NUM_LAYERS = 2  # number of layers in attention decoder cell

