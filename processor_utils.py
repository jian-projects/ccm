import random, torch, math, os
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import Adam, AdamW, SGD
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

def set_rng_seed(rng_seed: int = None, random: bool = True, numpy: bool = True,
                pytorch: bool = True, deterministic: bool = True):
    """
    设置模块的随机数种子。由于pytorch还存在cudnn导致的非deterministic的运行，所以一些情况下可能即使seed一样，结果也不一致
        需要在fitlog.commit()或fitlog.set_log_dir()之后运行才会记录该rng_seed到log中
        
    :param int rng_seed: 将这些模块的随机数设置到多少，默认为随机生成一个。
    :param bool, random: 是否将python自带的random模块的seed设置为rng_seed.
    :param bool, numpy: 是否将numpy的seed设置为rng_seed.
    :param bool, pytorch: 是否将pytorch的seed设置为rng_seed(设置torch.manual_seed和torch.cuda.manual_seed_all).
    :param bool, deterministic: 是否将pytorch的torch.backends.cudnn.deterministic设置为True
    """
    if rng_seed is None:
        import time
        import math
        rng_seed = int(math.modf(time.time())[0] * 1000000)
    if random:
        import random
        random.seed(rng_seed)
    if numpy:
        try:
            import numpy
            numpy.random.seed(rng_seed)
        except:
            pass
    if pytorch:
        try:
            import torch
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed(rng_seed)
            torch.cuda.manual_seed_all(rng_seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
        except:
            pass
    os.environ['PYTHONHASHSEED'] = str(rng_seed)  # 为了禁止hash随机化，使得实验可复现。
    return rng_seed

def random_parameters(args):
    # 固定参数
    if not args.train['seed_change']:
        args.logger['process'].warning("loading model and dataset, waiting ... ")
        lr, lr_pre = args.train['learning_rate'], args.train['learning_rate_pre']
        bz, dr, seed = args.train['batch_size'], args.model['drop_rate'], args.train['seed']
        args.logger['process'].warning(f"lr_{lr}, lr_pre_{lr_pre}, bz_{bz}, dr_{dr}, seed_{seed}")
        return args
    
    params = {
        'lr': [1e-3,5e-3,1e-2], #[0.03, 0.05, 0.08, 0.1, 0.2], 
        'bz': [16,32,64], # [16, 32, 64], 
        'dp': [0.1,0.3,0.5], 
    }
    args.train['learning_rate'] = random.choice(params['lr'])

    if 'AdamW' in args.model['optim_sched'][0]:
        scale = args.model['scale']
        params = {
            'lr': [1e-5,3e-5,5e-5,8e-5], 
            'bz': [16, 32] if scale=='base' else [8, 16, 32], 
            'dp': [0.1,0.3,0.5], 
        }
        args.train['epochs'] = 20 if scale=='base' else 10
        #args.params_train.epochs = random.randint(15,30)
        args.train['early_stop'] = 3 if scale=='base' else 2
        args.train['learning_rate_pre'] = random.choice(params['lr'])

    args.model['drop_rate'] = random.choice(params['dp'])
    args.train['seed'] += random.randint(-1000,1000)
    #args.params_train.seed = 225
    args.train['batch_size'] = int(random.choice(params['bz']))
    args.train['stop'] = 0 # 重新设置不停止

    lr, lr_pre = args.train['learning_rate'], args.train['learning_rate_pre']
    bz, dr, seed = args.train['batch_size'], args.model['drop_rate'], args.train['seed']
    args.logger['process'].warning(f"change: lr_{lr}, lr_pre_{lr_pre}, bz_{bz}, dr_{dr}, seed_{seed}")

    return args

def totally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    print(n_params)
    
    return n_params

def init_weight(model, method='xavier_uniform_'):
    if method == 'xavier_uniform_': fc = torch.nn.init.xavier_uniform_
    if method == 'xavier_normal_':  fc = torch.nn.init.xavier_normal_
    if method == 'orthogonal_':     fc = torch.nn.init.orthogonal_

    ## 非 plm 模型参数初始化
    for name, param in model.named_parameters():
        if 'plm' not in name: # 跳过 plm 模型参数
            if param.requires_grad:
                if len(param.shape) > 1: fc(param) # 参数维度大于 1
                else: 
                    stdv = 1. / math.sqrt(param.shape[0])
                    torch.nn.init.uniform_(param, a=-stdv, b=stdv)

def get_scheduler(args, optimizer, iter_total, method=None):
    scheduler = None
    if method is None: method = args.model['optim_sched'][-1]

    warmup_ratio = args.train['warmup_ratio']
    if 'linear' in method:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=iter_total*warmup_ratio, 
            num_training_steps=iter_total
        )
    if 'cosine' in method:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_ratio*iter_total, 
            num_training_steps=iter_total
        )

    return scheduler

def get_optimizer(args, model, methods=None):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if methods is None: methods = args.model['optim_sched']

    lr, lr_pre = args.train['learning_rate'], args.train['learning_rate_pre']
    weight_decay, adam_epsilon, l2reg = args.train['weight_decay'], args.train['adam_epsilon'], args.train['l2reg']

    no_decay = ['bias', 'LayerNorm.weight']
    if 'AdamW_' in methods:
        plm_params = list(map(id, model.plm_model.parameters()))
        model_params, warmup_params = [], []
        for name, model_param in model.named_parameters():
            weight_decay_ = 0 if any(nd in name for nd in no_decay) else weight_decay 
            lr_ = lr_pre if id(model_param) in plm_params else lr

            model_params.append({'params': model_param, 'lr': lr_, 'weight_decay': weight_decay_})
            warmup_params.append({'params': model_param, 'lr': lr_/4 if id(model_param) in plm_params else lr_, 'weight_decay': weight_decay_})
        
        model_params = sorted(model_params, key=lambda x: x['lr'])
        optimizer = AdamW(model_params)

    if 'AdamW' in methods:
        model_params = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0},
        ]
        optimizer = AdamW(model_params, lr=lr_pre, eps=adam_epsilon)
    
    if 'Adam' in methods: 
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(model_params, lr=lr, weight_decay=l2reg)

    if 'SGD' in methods:
        model_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = SGD(model_params, lr=lr, weight_decay=l2reg)

    return optimizer

def get_metric(results, dataset):
    if dataset.task == 'cls':
        labels, preds, total_loss = [], [], 0
        for rec in results:
            labels.extend(rec['labels'].cpu().numpy())
            preds.extend(rec['preds'].cpu().numpy())
            total_loss += rec['loss'].item()*len(rec['labels'])
        
        output = {}
        for item in dataset.metrics:
            if 'acc' in item:
                output[item] = round(accuracy_score(labels, preds), 4)
            if '_f1' in item:
                output[item] = round(f1_score(labels, preds, labels=dataset.lab_range, average=item.split('_')[0]), 4)
        output['loss'] = round(total_loss/len(labels), 3)

        return output

    if dataset.task == 'seg':
        total, total_dice, total_iou, total_loss = 0,0,0,0
        for rec in results:
            bz = rec['labels'].shape[0]
            total_dice += rec['dice_fn'](rec['logits'], rec['labels'].float()).item() * bz
            total_iou += rec['iou_fn'](rec['labels'], rec['logits']) * bz
            total_loss += rec['loss'].item() * bz
            total += bz
        
        return {
            'loss': round(total_loss/total, 2),
            'dice': round(total_dice/total, 2),
            'iou':  round(total_iou/total, 2),
        }