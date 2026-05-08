import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 项目根目录（本文件位于 config/ 下，向上退一级）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'params', 'vit_model.pt')
MLP1_PATH = os.path.join(PROJECT_ROOT, 'params', 'mlp1.pth')
TOK_EMBEDDING_PATH = os.path.join(PROJECT_ROOT, 'params', 'token_embedding.pth')
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'InternVL')
NORM_PARAMS_PATH = os.path.join(PROJECT_ROOT, 'params', 'gauss_norm_mu_sigma.pth')
NORM_TOK_EMBEDDING_PATH = os.path.join(PROJECT_ROOT, 'params', 'gauss_norm.pth')
NEW_1000_TOK_EMBEDDING_PATH = os.path.join(PROJECT_ROOT, 'params', 'new1000_token_embedding.pth')
INTERNVL_PATH = os.path.join(PROJECT_ROOT, 'InternVL')


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SEED=42

# 训练配置
BATCH_SIZE = 256
USE_WARMUP=False
LR = 1e-4  # original 1e-4
WEIGHT_DECAY = 1e-5
WARMUP_STEPS = 2000 # *4 = total training steps
NUM_EPOCHS = 13
NUM_WORKERS = 4
TRAIN_INTER = 10
VAL_INTER = 500
DOWNSAMPLE_RATIO = 0.5
NUM_LAYERS=4
GRAD_ACCU = 1
MODEL_NAME = 'PERCEIVER'



# 数据路径

TRAIN_DATA_PATH = ""
VAL_DATA_PATH = ''
TEST_DATA_PATH = ''
TRAIN_RATIO = 1#0.556 #0.02
VAL_RATIO = 0.2#0.1

# 36000 steps 8 cards 20 epochs, ~ 0.52 data ratio

# LOGS andSAVE_NAME
# 每一次跑新的实验切记一定需要修改！！！！
LOG_NAME = ''
SAVE_NAME = LOG_NAME+'.pth'

# DDP
WORLD_SIZE = torch.cuda.device_count()
# 如果我们要加载训练一半的模型，两个都不能是none!!
# LOAD CHECKPOINT AND RESUME TRAINING
# PERCEIVER_CHECKPOINT ="/home/luoyx/InternVL/CalliReader/params/perceiver_4_n01_1e-4_new.pth"
# RESUME = 26500
PERCEIVER_CHECKPOINT ='./params/callialign.pth'
RESUME = 50000
ORDERFORMER_CHECKPOINT='./params/orderformer.pth'
YOLO_CHECKPOINT="./params/best.pt"