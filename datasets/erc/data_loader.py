import json, torch, os
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

max_seq_lens = {'meld': 128, 'iec': 128, 'emn': 128, 'ddg': 128}


class DataLoader_ERC(Dataset):
    def __init__(self, dataset, d_type='multi', desc='train') -> None:
        self.d_type = d_type
        self.samples = dataset.datas['data'][desc]
        self.batch_cols = dataset.batch_cols
        self.tokenizer_ = dataset.tokenizer_
        self.tokenizer = dataset.tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample, output = self.samples[idx], {}
        for col, pad in self.keys.items():
            if col == 'audio':
                wav, _sr = sf.read(sample['audio'])
                output[col] = torch.tensor(wav.astype(np.float32))[0:160000]
            elif col == 'label':
                output[col] = torch.tensor(self.ltoi[sample[col]])
            else:
                output[col] = sample[col]
        return output


class ERCDataset_Multi(Dataset):
    def __init__(self, path, tokenizer=None, lower=False):
        self.path = path
        self.lower = lower
        self.name = ['erc', path.split('/')[-2]]
        self.container_init() # 初始化容器信息
        self.get_dataset()    # 解析数据集
        self.n_class = len(self.tokenizer_['labels']['ltoi']) 
        self.max_seq_len = max_seq_lens[self.name[-1]] # 最大句子长度
        self.type = 'cls' # 分类任务

    def container_init(self, only='all'):
        # 初始化数据集要保存的内容
        self.info = {
            'num_conv': {'train': 0, 'valid': 0, 'test': 0},              # number of convs
            'num_conv_speaker': {'train': [], 'valid': [], 'test': []},   # number of speakers in each conv
            'num_conv_utt': {'train': [], 'valid': [], 'test': []},       # number of utts in each conv
            'num_conv_utt_token': {'train': [], 'valid': [], 'test': []}, # number of tokens in each utt

            'num_samp': {'train': 0, 'valid': 0, 'test': 0},            # number of reconstructed samples
            'num_samp_token': {'train': [], 'valid': [], 'test': []},   # number of tokens in each sample
            'emotion_category': {},                                     # n_class
        }
        
        # 映射字典
        path_tokenizer_ = self.path + 'tokenizer_'
        if os.path.exists(path_tokenizer_):
            self.tokenizer_ = torch.load(path_tokenizer_)
            self.path_tokenizer_ = None
        else:
            self.tokenizer_ = {
                'labels': { 'ltoi': {}, 'itol': {}, 'count': {}},   # label 字典
                'speakers': { 'stoi': {}, 'itos': {}, 'count': {}}, # speaker 字典
            }
            self.path_tokenizer_ = path_tokenizer_

        # 数据集
        self.datas = {'data': {}, 'loader': {}}

    def speaker_label(self, speakers, labels):
        if self.path_tokenizer_ is None: return -1 # 已经加载好了

        # 记录speaker信息
        for speaker in speakers:
            if speaker not in self.tokenizer_['speakers']['stoi']: # 尚未记录
                self.tokenizer_['speakers']['stoi'][speaker] = len(self.tokenizer_['speakers']['stoi'])
                self.tokenizer_['speakers']['itos'][len(self.tokenizer_['speakers']['itos'])] = speaker
                self.tokenizer_['speakers']['count'][speaker] = 1
            self.tokenizer_['speakers']['count'][speaker] += 1

        # 记录label信息
        for label in labels:
            if label is None: continue
            if label not in self.tokenizer_['labels']['ltoi']:
                self.tokenizer_['labels']['ltoi'][label] = len(self.tokenizer_['labels']['ltoi'])
                self.tokenizer_['labels']['itol'][len(self.tokenizer_['labels']['itol'])] = label
                self.tokenizer_['labels']['class'].append(label)
                self.tokenizer_['labels']['count'][label] = 1
            self.tokenizer_['labels']['count'][label] += 1

    def get_dataset(self):
        for desc in ['train', 'valid', 'test']:
            raw_path = f'{self.path}/{desc}.raw.json'
            with open(raw_path, 'r', encoding='utf-8') as fp:
                raw_convs, convs = json.load(fp), []
            self.info['num_conv'][desc] = len(raw_convs)

            for ci, r_conv in enumerate(raw_convs):
                txts, spks, labs = [], [], []
                for utt in r_conv:
                    txt, spk, lab = utt['text'].strip(), utt['speaker'].strip(), utt.get('label')
                    txts.append(txt)
                    spks.append(spk)
                    labs.append(lab)

                self.speaker_label(spks, labs) # tokenizer_ (speakers/labels)
                convs.append({
                    'idx': len(convs),
                    'texts': txts,
                    'speakers': spks,
                    'emotions': labs,
                })
            self.datas['data'][desc] = convs
            
    # def vector_truncate(self, embedding, truncate='tail'):
    #     input_ids, attention_mask = embedding['input_ids'], embedding['attention_mask']
    #     cur_max_seq_len = max(torch.sum(attention_mask, dim=-1)) # 当前最大句子长度
    #     if cur_max_seq_len > self.max_seq_len:
    #         if truncate == 'tail': # 截断后面的
    #             temp = input_ids[:,0:self.max_seq_len]; temp[:, self.max_seq_len] = input_ids[:, -1]
    #             input_ids = temp
    #             attention_mask = attention_mask[:,0:self.max_seq_len]
    #         if truncate == 'first':
    #             temp = input_ids[:,-(self.max_seq_len):]; temp[:, 0] = input_ids[:, 0]
    #             input_ids = temp
    #             attention_mask = attention_mask[:,-(self.max_seq_len):]
    #         cur_max_seq_len = max(torch.sum(attention_mask, dim=-1))
    #     if truncate: assert cur_max_seq_len <= self.max_seq_len
    #     return input_ids, attention_mask

    # def refine_tokenizer(self, tokenizer):
    #     for token in self.tokens_add:
    #         tokenizer.add_tokens(token)
    #     return tokenizer

    # def get_vector(self, tokenizer, truncate='tail', only=None):
    #     self.tokenizer = tokenizer
    #     speaker_fn, label_fn = self.tokenizer_['speakers']['stoi'], self.tokenizer_['labels']['ltoi']
    #     for desc, dialogs in self.datas['text'].items():
    #         if only is not None and desc!=only: continue
    #         dialogs_embed = []
    #         for dialog in dialogs:
    #             embedding = tokenizer(dialog['texts'], max_length=self.max_seq_len, padding='max_length', return_tensors='pt')
    #             input_ids, attention_mask = self.vector_truncate(embedding, truncate=truncate)
    #             speakers = [speaker_fn[speaker] for speaker in dialog['speakers']]
    #             labels = [label_fn[label] for label in dialog['labels']]
    #             dialog_embed = {
    #                 'index': dialog['index'],
    #                 'input_ids': input_ids,
    #                 'attention_mask': attention_mask,
    #                 'speakers': torch.tensor(speakers),
    #                 'labels': torch.tensor(labels),
    #             }
    #             dialogs_embed.append(dialog_embed)

    #         self.info['total_samples_num'][desc] = len(dialogs_embed)
    #         self.info['max_token_num'][desc] = max([sample['attention_mask'].shape[-1] for sample in dialogs_embed])
    #         self.datas['vector'][desc] = dialogs_embed

    def get_dataloader(self, batch_size, shuffle=None, only=None):
        if shuffle is None:
            shuffle = {'train': True, 'valid': False, 'test': False}

        dataloader = {}
        for desc, data_embed in self.datas['data'].items():
            if only is not None and desc!=only: continue
            dataloader[desc] = DataLoader(dataset=data_embed, batch_size=batch_size, shuffle=shuffle[desc], collate_fn=self.collate_fn)
            
        return dataloader

    # def collate_fn(self, dialogs):
    #     max_token_num = max([max(dialog['attention_mask'].sum(dim=-1)) for dialog in dialogs])
    #     ## 获取 batch
    #     inputs = {}
    #     for col, pad in self.batch_cols.items():
    #         if 'index' in col: 
    #             temp = torch.tensor([dialog[col] for dialog in dialogs])
    #         if 'ids' in col or 'mask' in col:
    #             temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)[:,:,0:max_token_num]
    #         if 'speakers' in col or 'labels' in col:
    #             temp = pad_sequence([dialog[col] for dialog in dialogs], batch_first=True, padding_value=pad)
    #         inputs[col] = temp

    #     return inputs


class ERCDataset_Single(ERCDataset_Multi):
    def get_vector(self, args=None, tokenizer=None, method='tail', only=None):
        speaker_fn, label_fn = self.speakers['ntoi'], self.labels['ltoi']
        if args.anonymity: 
            tokenizer = self.refine_tokenizer(tokenizer) # 更新字典
            speaker_fn = self.speakers['atoi']
            
        self.args, self.tokenizer = args, tokenizer
        for desc, data in self.datas['text'].items():
            if only is not None and desc!=only: continue
            data_embed = []
            for item in data:
                embedding = tokenizer(item['text'], return_tensors='pt')
                input_ids, attention_mask = self.vector_truncate(embedding, method='first')
                speaker, label = speaker_fn[item['speaker']], label_fn[item['label']]
                item_embed = {
                    'index': item['index'],
                    'input_ids': input_ids.squeeze(dim=0),
                    'attention_mask': attention_mask.squeeze(dim=0),
                    'speaker': torch.tensor(speaker),
                    'label': torch.tensor(label),
                }
                data_embed.append(item_embed)

            self.datas['vector'][desc] = data_embed

    def collate_fn(self, samples):
        ## 获取 batch
        inputs = {}
        for col, pad in self.batch_cols.items():
            if 'ids' in col or 'mask' in col:  
                inputs[col] = pad_sequence([sample[col] for sample in samples], batch_first=True, padding_value=pad)
            else: 
                inputs[col] = torch.tensor([sample[col] for sample in samples])

        return inputs


def get_specific_dataset(args, d_type='multi'):
    ## 1. 导入数据
    data_path = args.file['data'] + f"{args.train['tasks'][1]}/"
    if d_type == 'multi': 
        dataset = ERCDataset_Multi(data_path, lower=True)
        dataset.batch_cols = {'idx': -1, 'texts': -1, 'speakers': -1, 'labels': -1 }
    else: 
        dataset = ERCDataset_Single(data_path, lower=True)
        dataset.batch_cols = {'idx': -1, 'texts': -1, 'speakers': -1, 'labels': -1 }

    dataset.tokenizer = AutoTokenizer.from_pretrained(args.model['plm'])
    dataset.shuffle = {'train': True, 'valid': False, 'test': False}
    for desc, data in dataset.datas['data'].items():
        dataset.datas['data'][desc] = DataLoader_ERC(
            dataset,
            d_type=d_type,
            desc=desc
        )
    dataset.task = 'cls'

    return dataset


# if __name__ == "__main__":
#     path = f'./datasets/erc/meld/'
#     dataset = ERCDataset_Multi(path, lower=True)
#     tokens = []
#     for desc, samples in dataset.datas['text'].items():
#         for sample in samples:
#             for text in sample['texts']:
#                 tokens.extend(text.split(' '))
#     tokenizer = get_tokenizer(path=path+'glove.tokenizer', tokens=tokens)
#     dataset.get_vector(tokenizer, truncate=None)

#     input()