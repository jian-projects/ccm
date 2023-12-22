import argparse, torch, os, sys
from datetime import datetime
from loguru import logger

# 59.77.7.19
PLM_DIR = '/home/jzq/My_Codes/Pretrained_Models/'
# DATA_DIR = '/home/jzq/My_Codes/Datasets/Classification/'
DATA_DIR = './datasets/'
SAVE_DIR = './checkpoints/'


def log_show(desc, args=None):
    if 'load' in desc: ## 0. 加载参数显式
        lr, lr_pre = args.params_train.learning_rate, args.params_train.learning_rate_pre
        bz, dr, seed = args.params_train.batch_size, args.params_model.drop_rate, args.params_train.seed
        desc_training_params = f"lr_{lr}, lr_pre_{lr_pre}, bz_{bz}, dr_{dr}, seed_{seed}"
        if 'change' not in desc: args.logger.info(desc_training_params)
        else: args.logger.info('change: ' + desc_training_params)
    if 'eval' in desc: ## 0. 加载参数显式
        args.logger.info("***** evaluating_on_{}, waiting *****".format(desc[-1]))

def config(tasks=['absa', 'lap'], models=['tnet', None]):
    args = argparse.ArgumentParser().parse_args()

    ## parameters for output
    args.file = {
        'plm_dir': PLM_DIR,
        'data_dir': DATA_DIR + f"{tasks[0]}/",
        'save_dir': f'{SAVE_DIR}/{tasks[0]}/{tasks[1]}/',

        'log': f'./logs/{tasks[0]}/{tasks[1]}/',
        'record': f'./records/{tasks[0]}/{tasks[1]}/',
    }
    sys.path.append(args.file['data_dir']) # 添加数据路径

    ## parameters for training
    args.train = {
        'tasks': tasks,
        'show': False,

        'stop': 0,
        'epochs': 64,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'device_ids': [0],
        'do_test': True,
        'do_valid': True,
        'do_train': True,
        'early_stop': 8,
        'save_model': False,
        'log_step_rate': 1, # 每个epoch将进行评价次数

        'seed': 2023,
        'l2reg': 0.01,
        'data_rate': 1.0,
        'batch_size': 64,
        'seed_change': True,
        'warmup_ratio': 0.3,
        'weight_decay': 1e-3,
        'adam_epsilon': 1e-8,
        'max_grad_norm': 5.0,
        'learning_rate': 1e-3,
        'learning_rate_pre': 1e-5,
    }

    ## parameters for model
    args.model = {
        'name': models[0],
        'backbone': models[1] if len(models)>1 else None,
        'drop_rate': 0.3,
    }

    ## logging
    logger.remove() # 不要在控制台输出日志
    handler_id = logger.add(sys.stdout, level="WARNING") # WARNING 级别以上的日志输出到控制台

    logDir = os.path.expanduser(args.file['log']+datetime.now().strftime("%Y%m%d_%H%M%S")) # 日志文件夹
    if not os.path.exists(logDir): os.makedirs(logDir) # 创建日志文件夹

    logger.add(os.path.join(logDir,'loss.log'), filter=lambda record: record["extra"].get("name")=="loss") # 添加配置文件 loss
    logger.add(os.path.join(logDir,'metric.log'), filter=lambda record: record["extra"].get("name")=="metric") # 添加配置文件 metric
    logger.add(os.path.join(logDir,'process.log'), filter=lambda record: record["extra"].get("name")=="process") # 添加配置文件 metric
    args.logger= {
        'loss': logger.bind(name='loss'), 
        'metric': logger.bind(name='metric'),
        'process': logger.bind(name='process'),  
    } # 日志记录器
    
    return args