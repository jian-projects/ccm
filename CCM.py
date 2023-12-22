import torch, os, copy, math, json, sys
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset

from processor_utils import set_rng_seed
from utils.gnn import GAT
from utils.lstm import DynamicLSTM

# config 中已经添加路径了
from data_loader import get_specific_dataset, ERCDataset_Multi


"""
| dataset     | meld  | iec   | emn   | ddg   |

| baseline    | 67.25 | 69.74 | 40.94 | 00.00 | # test
| performance | 65.11 | 00.00 | 00.00 | 00.00 |

"""
baselines = {
    'base': {'meld': 0., 'iec': 0., 'emn': 0., 'ddg': 0}, # valid
    'large': {'meld': 0., 'iec': 0., 'emn': 0., 'ddg': 0},
}

# flow_nums = {'meld': 4, 'emn': 5, 'iec': 3, 'ddg': 3}
max_seq_lens = {'meld':128, 'emn': 64, 'iec': 64, 'ddg': 160}

def visible_matrix(dialog):
    new_dialog = dialog['new_dialog']
    speakers = new_dialog['speakers']
    new_dialog['visible'], new_dialog['position'] = [], []
    for si, speaker in enumerate(speakers):
        temp_w, wei = [0]*len(speakers), 1
        temp_p, pos = [0]*len(speakers), 1
        # temp[si] = 1
        # 往回找，遇见 speaker 权重减小
        for i in list(range(0,si))[::-1]:
            if speakers[i] != speaker: # 与当前 speaker 不一样
                temp_w[i] = wei
                temp_p[i] = pos
            else: # 与当前 speaker 一样, 权重降低
                temp_w[i] = wei
                temp_p[i] = pos
                pos += 1
                #break
                # if speakers[i] != speakers[i-1]:
                #     wei *= 0.5
                # temp[i] = wei
            
        #temp = [v/sum(temp) for v in temp] # 归一化
        new_dialog['visible'].append(temp_w)
        new_dialog['position'].append(temp_p)

class DataLoader_CCM(Dataset):
    def __init__(self, dataset, d_type='multi', desc='train') -> None:
        self.d_type = d_type
        self.dataset = dataset
        self.samples = dataset.datas['data'][desc]
        self.batch_cols = dataset.batch_cols

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class ERCDataset_CCM(ERCDataset_Multi):
    def emotion_static(self, dialogs):
        ltoi = self.tokenizer_['labels']['ltoi']
        pi, A = [0]*len(ltoi), [[0]*len(ltoi) for _ in range(len(ltoi))]
        for dialog in dialogs:
            labs = dialog['emotions']
            for i, lab in enumerate(labs):
                if lab is None: 
                    continue
                if i==len(labs)-1: continue
                if i == 0: 
                    pi[ltoi[lab]] += 1.0
                    pre_lab = lab
                else:
                    A[ltoi[pre_lab]][ltoi[lab]] += 1.0
                    pre_lab = lab
        
        # 归一化
        new_pi = F.softmax(torch.tensor([v/sum(pi) for v in pi]))
        new_A = []
        for a in A: 
            temp = F.softmax(torch.tensor([v/sum(a) for v in a]))
            new_A.append(temp)
        
        self.info['hmm_P'] = new_pi
        self.info['hmm_A'] = torch.stack(new_A)

    def get_vector(self, truncate=None, only=None):
        tokenizer = self.tokenizer
        sep_token, mask_token = tokenizer.sep_token, tokenizer.mask_token
        for stage, convs in self.datas['data'].items():
            if only is not None and stage!=only: continue
            for conv in tqdm(convs):
                conv['labels'], conv['prompts'], conv['clique'] = [], [], []
                conv['input_ids'], conv['attention_mask'] = [], []
                for ui in range(len(conv['texts'])):
                    # label id:
                    conv['labels'].append(self.tokenizer_['labels']['ltoi'][conv['emotions'][ui]] if conv['emotions'][ui] is not None else -1)
                    # enhanced utterance & conversation clique
                    utt_clique = [0]*len(conv['texts']); utt_clique[ui] = 1; mark = [] # 当前 utterance 包含在clique
                    utt_prompt = conv['speakers'][ui]+': '+conv['texts'][ui]+f' {sep_token} '+conv['speakers'][ui]+f' displays {mask_token}'
                    for k in list(range(ui))[::-1]:
                        if conv['speakers'][ui] not in mark: utt_clique[k] = 1 # 当前 speaker 前一句话尚未触及
                        if len(mark)==0 and conv['speakers'][k]==conv['speakers'][ui]: pass
                        else: mark.append(conv['speakers'][k])
                        utt_prompt = conv['speakers'][k]+': '+conv['texts'][k]+f' {sep_token} ' + utt_prompt
                    conv['prompts'].append(utt_prompt)
                    conv['clique'].append(utt_clique)
                    
                    # tokenizer 
                    conv['input_ids'].append(torch.tensor([tokenizer.cls_token_id]+self.tokenizer(utt_prompt, add_special_tokens=False)['input_ids'][-self.max_seq_len:]))
                    conv['attention_mask'].append(torch.ones_like(conv['input_ids'][-1]))

                conv['input_ids'] = pad_sequence(conv['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id)
                conv['attention_mask'] = pad_sequence(conv['attention_mask'], batch_first=True, padding_value=tokenizer.pad_token_id)
                conv['labels'], conv['clique'] = torch.tensor(conv['labels']), torch.tensor(conv['clique'])

                # conv = visible_matrix(conv)

    def collate_fn(self, samples):
        bz = len(samples)
        return {
            'idx': [torch.tensor(samples[i]['idx']) for i in range(bz)],
            'input_ids': [samples[i]['input_ids'] for i in range(bz)],
            'attention_mask': [samples[i]['attention_mask'] for i in range(bz)],
            'clique': [samples[i]['clique'] for i in range(bz)],
            'labels': [torch.tensor(samples[i]['labels']) for i in range(bz)],
            # 'visible': [torch.tensor(dialogs[i]['visible']) for i in range(bz)],
            # 'position': [torch.tensor(dialogs[i]['position']) for i in range(bz)],
        }
    

def config_for_model(args):
    scale = args.model['scale']
    args.model['plm'] = args.file['plm_dir'] + f'roberta-{scale}'
    args.model['baseline'] = baselines[scale][args.train['tasks'][1]]
    args.model['tokenizer'] = None
    args.model['optim_sched'] = ['AdamW_', 'cosine']
    #args.model['optim_sched'] = ['AdamW_', 'linear']

    args.model['hmm'] = 1
    args.model['gat'] = 1
    args.model['rnn'] = 1

    return args
             
def import_model(args):
    ## 1. 更新参数
    args = config_for_model(args) # 添加模型参数, 获取任务数据集
    set_rng_seed(args.train['seed'])
    
    ## 2. 导入数据
    data_path = args.file['data_dir'] + f"{args.train['tasks'][1]}/"
    dataset = ERCDataset_CCM(data_path, lower=True)
    dataset.tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
    dataset.get_vector()
    dataset.emotion_static(dataset.datas['data']['train']) # 统计 HMM
    dataset.batch_cols = {
        'index': -1,
        'input_ids': dataset.tokenizer.pad_token_id,
        # 'attention_mask': 0,
        'labels': -1,
        'visible': 0,
        'position': 0,
    }
    dataset.shuffle = {'train': True, 'valid': False, 'test': False}
    for desc, data in dataset.datas['data'].items():
        dataset.datas['data'][desc] = DataLoader_CCM(
            dataset,
            d_type='multi',
            desc=desc
        )
    dataset.task = 'cls'
    
    # 3. 模型
    model = CCM(
        args=args,
        dataset=dataset,
        plm=args.model['plm'],
    )

    return model, dataset
    

class PoolerAll(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states # [:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class CCM(nn.Module):
    def __init__(self, args, dataset, plm=None):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.n_class = dataset.n_class
        self.mask_token_id = dataset.tokenizer.mask_token_id

        self.plm_model = AutoModel.from_pretrained(plm, local_files_only=False) 
        self.plm_model.pooler_all = PoolerAll(self.plm_model.config)
        self.hidden_dim = self.plm_model.config.hidden_size   

        self.gat = GAT(
            nfeat=self.hidden_dim, 
            nhid=self.hidden_dim//8, 
            nclass=self.hidden_dim, 
            dropout=0.3, 
            nheads=8, 
            alpha=0.2)
        self.dropout = nn.Dropout(args.model['drop_rate'])
        self.classifier = nn.Linear(self.hidden_dim, self.n_class)
        self.linear_0 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.linear_1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_cat = nn.Linear(self.hidden_dim*2, self.hidden_dim) # 拼接 history utt -> 恢复维度
        self.linear_cat_1 = nn.Linear(self.hidden_dim*2, self.hidden_dim) # 拼接 rnn gat -> 恢复维度
        self.loss_ce = CrossEntropyLoss(ignore_index=-1)
        # self.loss_sce = LabelSmoothSoftmaxCEV1(lb_smooth=None, reduction='weight', ignore_index=-1)

        self.hmm_P = (torch.ones(self.n_class)*1/self.n_class).unsqueeze(dim=1)
        self.hmm_A = dataset.info['hmm_A'] # [nn.Sequential(nn.Linear(self.hidden_dim, 7), nn.Softmax()).to('cuda') for _ in range(7)]
        self.hmm_B = nn.Sequential(
            self.dropout,
            self.classifier, #nn.Linear(self.hidden_dim, n_class), 
            nn.Softmax()
        ) # 需要归一化

        self.print_trainable_parameters(self) # 参与训练的参数
   
    def print_trainable_parameters(self, model):
            """
            Prints the number of trainable parameters in the model.
            """
            trainable_params = 0
            all_param = 0
            for _, param in model.named_parameters():
                num_params = param.numel()
                # if using DS Zero 3 and the weights are initialized empty
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel

                all_param += num_params
                if param.requires_grad:
                    trainable_params += num_params
            print(
                f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
            )

    def encode(self, inputs):
        plm_out = self.plm_model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = self.plm_model.pooler_all(plm_out.last_hidden_state)

        # 取出 mask 表示
        mask_bool = inputs['input_ids'] == self.mask_token_id
        hidden_states_mask, loss_mlm = hidden_states[mask_bool], 0

        return hidden_states[:, 0], hidden_states_mask

    def hmm(self, obv):
        log_hmm = []
        for fi, fea in enumerate(obv):
            if fi == 0:
                bb = self.hmm_B(fea).reshape_as(self.hmm_P)
                p = self.hmm_P.type_as(fea)*bb
                p = p*1/sum(p)
            else:
                p = torch.mm(self.hmm_A.type_as(fea).T, log_hmm[-1].T) * self.hmm_B(fea).reshape_as(self.hmm_P)
                p *= 1/sum(p)
            log_hmm.append(p.transpose(1,0))

        return log_hmm

    def forward(self, inputs, stage='train'):
        bz = len(inputs['input_ids'])
        labels_bz, log_gat_bz, log_rnn_bz, log_cls_bz, log_hmm_bz = [], [], [], [], []
        for i in range(bz):
            # encoding
            input_tmp = { }
            for k, v in inputs.items():
                input_tmp[k] = v[i].to(self.args.train['device'])
            fea_cls, fea_emo = self.encode(input_tmp)

            # clique batch
            # fea_cls_contextual, fea_emo_contextual = torch.clone(fea_cls), torch.clone(fea_emo) # 拼接了上下文的表示
            utts = {'clique': [], 'labels': [], 'fea_cls': [], 'fea_emo': []}
            for j in range(len(input_tmp['clique'])):
                utts['clique'].append(input_tmp['clique'][j])
                utts['labels'].append(input_tmp['labels'][j])
                utts['fea_emo'].append(fea_emo[utts['clique'][-1].bool()])
                utts['fea_cls'].append(fea_cls[utts['clique'][-1].bool()])

                # fea_cls_contextual[j] = self.linear_cat(torch.cat([fea_cls[utts['clique'][-1].bool()].mean(dim=0), fea_cls[j]], dim=-1))
                # utts['fea_cls'].append(fea_cls_contextual[utts['clique'][-1].bool()])
                # fea_emo_contextual[j] = self.linear_cat(torch.cat([fea_emo[utts['clique'][-1].bool()].mean(dim=0), fea_emo[j]], dim=-1))
                # utts['fea_emo'].append(fea_emo_contextual[utts['clique'][-1].bool()])

            utts['clique'] = torch.stack(utts['clique'])
            utts['fea_cls_'] = pad_sequence(utts['fea_cls'], batch_first=True, padding_value=self.dataset.tokenizer.pad_token_id)

            # gat
            if self.args.model['gat']:
                for _ in range(1):
                    fea_cls_gat = self.gat(fea_cls, input_tmp['clique']-torch.eye(len(fea_cls)).type_as(fea_cls))
                    fea_cls_gat = torch.cat([self.dropout(fea_cls)[0:1], fea_cls_gat[1:]]) # 对话 第一个utterance没有邻居
                    fea_cls_gat = self.linear_0(torch.cat([fea_cls, self.linear_1(fea_cls_gat)], dim=-1))
                    log_gat_bz.append(F.softmax(self.classifier(self.dropout(fea_cls_gat))))

            # hmm
            if self.args.model['hmm']:
                log_hmm = []
                for clique_b in utts['fea_emo']:
                    p = self.hmm(clique_b)
                    # log_hmm_add.extend(p)
                    # labels_hmm_add.extend(copy.deepcopy(utts['labels']))
                    log_hmm.append(p[-1])
                log_hmm_bz.append(torch.cat(log_hmm))
        
            labels_bz.append(input_tmp['labels'])
        
        labels = torch.cat(labels_bz)
        logits = torch.log((torch.cat(log_gat_bz, dim=0)+torch.cat(log_hmm_bz))/2)
        # logits = torch.log((torch.cat(log_gat_bz, dim=0)+torch.cat(log_rnn_bz, dim=0)+torch.cat(log_hmm_bz))/3)

        lab_mask = labels >= 0
        labels, logits = labels[lab_mask], logits[lab_mask]
        assert -1 not in labels

        lb_one_hot = torch.empty_like(logits).fill_(0).scatter_(1, labels.unsqueeze(1), 1).detach()
        loss = -torch.sum(logits * lb_one_hot, dim=1).mean()

        return {
            'loss':   loss,
            'logits': logits,
            'preds':  torch.argmax(logits, dim=-1).cpu(),
            'labels': labels,
        }