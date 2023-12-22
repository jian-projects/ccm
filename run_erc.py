import warnings, os, random, wandb
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 可用的GPU
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from config import config
from writer import JsonFile
from processor import Processor
from processor_utils import *

def get_model(args, model_name=None):
    if model_name is None: 
        model_name = [args.model['name'], args.model['backbone']]

    ## 框架模型
    if model_name[-1] is not None: 
        return None
    ## 非框架模型
    if 'ccm' in model_name: from CCM import import_model

    model, dataset = import_model(args)
    init_weight(model)

    return model, dataset

def run(args):
    args = random_parameters(args)
    model, dataset = get_model(args)
    if torch.cuda.device_count() > 1: # 指定卡/多卡 训练
        model = torch.nn.DataParallel(model, device_ids=args.train['device_ids'])

    if dataset.type == 'cls':
        from processor import Processor
    else: from processor_gen import Processor

    dataset.metrics = ['weighted_f1', 'micro_f1', 'accuracy']
    dataset.lab_range = list(range(dataset.n_class)) if dataset.name[-1]!='ddg' else list(range(1, dataset.n_class))
    processor = Processor(args, model, dataset)
    
    if args.train['inference']:
        processor.loadState()
        processor._evaluate(stage='test')
    else: result = processor._train()

    if args.train['wandb']: wandb.finish() 
    ## 2. 输出统计结果
    record = {
        'params': {
            'e':       args.train['epochs'],
            'es':      args.train['early_stop'],
            'lr':      args.train['learning_rate'],
            'lr_pre':  args.train['learning_rate_pre'],
            'bz':      args.train['batch_size'],
            'dr':      args.model['drop_rate'],
            'seed':    args.train['seed'],
        },
        'metric': {
            'stop':    result['epoch'],
            'tv_mf1':  result['valid']['weighted_f1'],
            'te_mf1':  result['test']['weighted_f1'],
        },
    }
    return record


if __name__ == '__main__':

    """
    tasks: 
        ddg: DailyDialog
        emn: EmoryNLP
        iec: IEMOCAP
        meld: MELD

    models: 
        dialogxl:
        daterc:
        cogbart: 
        spcl: EMNLP2022 (simcse、SCL、Curriculum Learning)
        emotionflow: ICASSP2022 (Roberta、CRF、Emotion Flow) (一个batch是一个dialog)

        emoTrans: Ours (simcse、Retrieval、SCL、Emotion Flow)
        ccm:  Ours (GAT、HMM)
    """
    args = config(tasks=['erc','meld'], models=['ccm', None])

    ## Parameters Settings
    args.model['scale'] = 'large'
    args.train['device_ids'] = [0]
    
    args.train['epochs'] = 6
    args.train['early_stop'] = 2
    args.train['batch_size'] = 2
    args.train['save_model'] = False
    args.train['log_step_rate'] = 1.0
    args.train['learning_rate'] = 5e-6
    args.train['learning_rate_pre'] = 5e-6

    args.model['drop_rate'] = 0.1

    args.train['inference'] = 0
    args.train['wandb'] = 0 # True
    if args.train['wandb']:
        wandb.init(
            project=args.train['task']+'_22',
            name=args.train['data'],
        )

    seeds = []
    ## Cycle Training
    if seeds: # 按指定 seed 执行
        recoed_path = f"{args.file['record']}{args.model['name']}_best.json"
        record_show = JsonFile(recoed_path, mode_w='a', delete=True)
        for seed in seeds:
            args.train['seed'] = seed
            args.train['seed_change'] = False
            record = run(args)
            record_show.write(record, space=False) 
    else: # 随机 seed 执行       
        recoed_path = f"{args.file['record']}{args.model['name']}_search.json"
        record_show = JsonFile(recoed_path, mode_w='a', delete=True)
        for c in range(100):
            args.train['seed'] = random.randint(1000,9999)+c
            args.train['seed_change'] = False
            record = run(args)
            record_show.write(record, space=False)