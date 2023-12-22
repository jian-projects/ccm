import torch, time, logging, json, os
import numpy as np
from tqdm import tqdm

from processor_utils import *
# from similarity import k_means

logging.basicConfig(
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
    )

class Processor():
    def __init__(self, args, model, dataset) -> None:
        self.args = args
        self.dataset = dataset
        self.model = model.to(args.train['device'])

        ## 0. 根据最新batch设置dataloader
        if self.dataset.datas['loader']: self.dataloader = self.dataset.datas['loader']
        else: self.dataloader = self.dataset.get_dataloader(args.train['batch_size'], shuffle=dataset.shuffle)
        self.log_step_rate = args.train['log_step_rate']*1

        ## 记录运行状态
        self.save_path = f"{args.file['save_dir']}/{args.model['name']}/"
        if not os.path.exists(self.save_path): 
            os.makedirs(self.save_path)

        self.logger_loss, self.logger_metric, self.logger_process = args.logger['loss'], args.logger['metric'], args.logger['process']
        self.metrics, self.log_show, self.global_step = dataset.metrics, args.train['show'], 1

        self.best_result = {
            'epoch': 0, 'times': 0, 'loss':  0,
            # metrics
            'train': {item: 0 for item in self.metrics},
            'valid': {item: 0 for item in self.metrics},
            'test':  {item: 0 for item in self.metrics},      
        }
        #if 'fw' not in args.params_model.name[0]: args.params_model.baseline = 1

    def epoch_deal(self, epoch=None, inputs=None, stage='init'):
        if stage == 'stop':
            if 'attack' in self.args.model and self.args.model['attack'] : ## ral attack
                # 统计聚类中心，判断聚类半径
                cluster_samps = {'cls': [], 'lab': []}
                for bi, batch in enumerate(self.dataloader['train']):
                    self.model.eval()
                    with torch.no_grad():
                        outs = self.model_calculate(batch, 'test') 
                        cluster_samps['cls'].extend(outs['clss'].detach().cpu().numpy())
                        cluster_samps['lab'].extend(outs['labels'].cpu().numpy())
                
                cluster_result = k_means(cluster_samps['cls'], n_clusters=self.dataset.n_class)


                radii = [] # 计算每个聚类的半径
                for i, center in enumerate(cluster_result.cluster_centers_):
                    index = [li for li,l in enumerate(cluster_result.labels_) if l==i]
                    idx_lab = [cluster_samps['lab'][idx] for idx in index]


                    dists = np.linalg.norm(data - center, axis=1)  # 计算每个点到中心的距离
                    radius = np.max(dists)  # 取最大距离作为半径
                    radii.append(radius)

                print(radii)


                labels, attacks = [], []
                for rec in inputs:
                    labels.extend(rec['labels'].cpu().numpy())
                    attacks.extend(rec['attack'])
                errors = [l!=a for l, a in zip(labels, attacks) if a!=-1]
                self.logger_process.warning(f"attack: {round(sum(errors)/len(errors), 4)}")


                

        # model_name = self.args.model['name']
        # if 'rcl' in model_name[0]:
        #     self.model.init_deal(epoch, self.dataset, self.params.device)

        # if 'cscl' in model_name[-1]:
        #     self.model.epoch_deal(epoch, self.dataset, self.params.device) # 更新检索信息

        # # if 'efcl' in model_name[-1]:
        # #     self.model.epoch_deal(epoch, self.dataset, self.params.device) # 更新检索信息

        # if 'spcl' in model_name:
        #     dataset = self.model.epoch_deal(epoch, self.dataset, self.params.device) # 更新训练数据
        #     ## 更新 dataset 相关信息; log_step\optimizer 都先不更新了
        #     dataset.datas['dataloader'] = dataset.get_dataloader(self.params.batch_size, shuffle=self.dataset.shuffle)
        #     self.dataloader = dataset.datas['dataloader']

        # if 'emotionflow' in model_name:
        #     self.optimizer, self.scheduler = self.model.optimizer_(self.params)
        #     # self.dataset.get_dataloader(self.params.batch_size, shuffle=self.dataset.shuffle)
        #     # self.dataloader = dataset.datas['dataloader']dataset

    def model_calculate(self, batch, stage):
        if not isinstance(batch['input_ids'], list):
            for key, val in batch.items(): 
                batch[key] = val.to(self.args.train['device'])
        outs = self.model(batch, stage) # 模型计算

        return outs

    def train_epoch(self, epoch, stage='train'):
        #if self.model.params.about == 'efcl': return self.train_epoch_(epoch, stage)

        log_step = int(len(self.dataloader['train']) / self.log_step_rate)
        model, args = self.model, self.args
        self.epoch_deal(epoch=epoch) # epoch开始前/后进行一些处理
        loss_epoch, results_epoch = [], []# 没有按 index 顺序
        torch.cuda.empty_cache()
        #for batch in tqdm(self.dataloader['train'], smoothing=0.05):
        for bi, batch in enumerate(self.dataloader['train']):
            # 数据计算
            model.train()      
            outs = self.model_calculate(batch, stage)    
            
            # 误差传播
            loss = outs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.train['max_grad_norm'])
            self.optimizer.step()
            if self.scheduler is not None: self.scheduler.step() 
            self.optimizer.zero_grad()        

            # 结果统计  
            self.global_step += 1
            results_epoch.append(outs)

            # 过程展示
            if self.global_step % log_step == 0:
                rec_valid = self._evaluate(stage='valid')
                self.update_metric(epoch, rec_valid, stage='valid')
        
            # if args.train['wandb'] and self.global_step % log_step//3 == 0:
            #     wandb.log({
            #         "train_acc": round(accuracy_score(labels, preds), 4), 
            #         "train_loss": loss,
            #         "valid_acc": self.best_result['valid']['accuracy']
            #         })

        self.epoch_deal(epoch=epoch, inputs=results_epoch) # epoch开始前/后进行一些处理
        return results_epoch

    def loadState(self, iter=0, bl_rate=0.9):
        start_e, args, model = 0, self.args, self.model
        iter_total = int(len(self.dataloader['train'])*args.train['epochs']) - iter
        self.optimizer = get_optimizer(args, model)
        self.scheduler = get_scheduler(args, self.optimizer, iter_total)

        # load checkpoint
        if self.args.train['inference']: # 
            checkpoint = torch.load(self.save_path+f'model.state')
            self.model.load_state_dict(checkpoint['net'])
            self.best_result = checkpoint['result']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        return start_e, args

    def _train(self):
        ## Initial Model and Processor
        start_e, args = self.loadState()

        ## Training epochs
        epochs = args.train['epochs']
        self.tqdm_epochs = tqdm(total=epochs, position=0)
        self.train_desc(epoch=-1) # initialize process bar
        for epoch in range(start_e, epochs):
            s_time = time.time()
            output_epoch = self.train_epoch(epoch) # train one epoch
            
            rec_train = get_metric(output_epoch, self.dataset)
            rec_train['time'] = round(time.time()-s_time, 3)
            self.update_metric(epoch, rec_train, stage='train')
            torch.cuda.empty_cache()
            self.train_desc(epoch, rec_train) # update process bar

            if self.train_stop(epoch): break # is early stop ?
            
        self.tqdm_epochs.close()
        self.epoch_deal(epoch=epoch, stage='stop') # 训练停止, 是否做些处理
        return self.best_result

    def _evaluate(self, stage='test'):
        ## 显示当前状态
        # if self.log_show: 
        #     self.logger_process.warning("***** evaluating_on_{}, waiting *****".format(stage))
        
        ## 计算预测结果
        results_epoch = []
        for bi, batch in enumerate(self.dataloader[stage]):
            self.model.eval()
            with torch.no_grad():
                outs = self.model_calculate(batch, stage) 
                results_epoch.append(outs)
        
        ## 计算评价指标
        score = get_metric(results_epoch, self.dataset)

        return score
    
    def update_metric(self, epoch, score, stage='valid'):
        args = self.args
        ## 按valid选择最佳模型
        if 'valid' in stage:
            if self.best_result['valid'][self.metrics[0]] < score[self.metrics[0]]:
                # 1. 更新验证集上最佳结果
                self.best_result['epoch'] = epoch
                for metric in self.metrics: self.best_result[stage][metric] = score[metric]

                # 2. logger 过程写入
                if self.log_show: 
                    self.logger_process.warning("update: {}".format(json.dumps(score)))
                self.logger_metric.info(f"{stage}_eval: " + json.dumps(score))

                # 3. 是否需要保存 checkpoint
                if args.train['save_model']:  
                    # torch.save(self.model, self.save_path)
                    state = {
                        'net': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'step': self.global_step,
                        'result': self.best_result,
                        'batch_size': args.train['batch_size'],
                    }
                    torch.save(state, self.save_path+f'model.state')

                # 4. 看看在测试集上的效果
                if args.train['do_test']:
                    rec_test = self._evaluate(stage='test')
                    self.update_metric(epoch, rec_test, stage='test')


        ## 记录test的结果
        if 'test' in stage:
            for metric in self.metrics: self.best_result[stage][metric] = score[metric]

            if self.log_show: 
                self.logger_process.warning("test: {}".format(json.dumps(score)))
            self.logger_metric.info('test_test_test: ' + json.dumps(score))

        ## 记录train的结果
        if 'train' in stage:
            self.best_result['loss'] = score['loss']
            self.best_result['times'] = max(self.best_result['times'], score['time'])
            for metric in self.metrics: self.best_result[stage][metric] = score[metric]

            if self.log_show: 
                self.logger_process.warning(f"train: {json.dumps(score)}")
            self.logger_metric.info('train: ' + json.dumps(score))
            self.logger_loss.info('train_loss: ' + str(score['loss']))


    def train_desc(self, epoch, outs=None):
        args, metrics, results = self.args, self.metrics, self.best_result

        epochs, model_name, data_name = args.train['epochs'], args.model['name'], self.dataset.name[-1]
        best_train, best_valid, best_test = results['train'][metrics[0]], results['valid'][metrics[0]], results['test'][metrics[0]]
        cur_total_loss = 0 if outs is None else outs['loss']
        consume_time = 0 if outs is None else outs['time']
        desc = f"epoch {epoch}/{epochs} ( {model_name} => {data_name}: {best_train}/{best_valid}/{best_test}, loss: {cur_total_loss}, time: {consume_time} )"

        self.tqdm_epochs.set_description(desc)
        if epoch>=0: self.tqdm_epochs.update()

    def train_stop(self, epoch=None):
        args = self.args

        # 0. 启动太差，重来
        # if self.best_result[self.metrics[0]]['valid'] < params_model.baseline/1.3:
        #     params.stop = 1
        # else:
        #     fw = open(f'{self.dataset.name[-1]}_search.txt', 'a', encoding='utf-8')
        #     desc_0 = f'lr_{params.learning_rate_pre}, bz_{params.batch_size}, dr_{params_model.drop_rate}, seed_{params.seed}'
        #     desc_1 = f'epochs_{params.epochs}, early_stop_{params.early_stop}, scale_{params_model.scale}'
        #     fw.write(f"{desc_1}, {desc_0}, score_{self.best_result[self.metrics[0]]['valid']} \n")
        #     fw.close()
        #     params.stop = 1

        # if self.best_result[self.metrics[0]]['valid'] < params_model.baseline:
        #     params.stop = 1

        # 1. 长期未更新了，增加评价次数
        early_threshold = epoch - self.best_result['epoch']
        if early_threshold >= args.train['early_stop']:
            return True

        # self.log_step_rate = (self.params.log_step_rate+early_threshold)/1.3 # for absa
        # self.log_step_rate = (args.train['log_step_rate']+early_threshold)/0.8