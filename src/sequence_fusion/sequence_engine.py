import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.sequence_fusion.utils.plotting import plot_metrics
from src.utils.metrics import AverageMeter



#
class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(FCNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.classifier = nn.Linear(hidden_sizes[-1], 2)
        fc_layers = []
        # Add input layer
        fc_layers.append(nn.Linear(input_size, hidden_sizes[0]))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(0.25))

        # Add hidden layers
        for i in range(len(hidden_sizes) - 1):
            fc_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(0.25))

        # Convert to sequential container
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        out = self.fc(x)
        pre = self.classifier(out)
        return out,pre
#








class Engine(object):
    def __init__(self, args, results_dir, fold):
        self.args = args
        self.results_dir = results_dir
        self.fold = fold
        self.best_score = 0
        self.best_epoch = 0
        self.filename_best = None

    def learning(self, model, train_loader, val_loader, test1_loader,criterion, optimizer, scheduler):
        if torch.cuda.is_available():
            model = model.cuda()

        if self.args.resume is not None:
            if os.path.isfile(self.args.resume):
                print("=> loading checkpoint '{}'".format(self.args.resume))
                checkpoint = torch.load(self.args.resume)
                self.best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint (score: {})".format(checkpoint['best_score']))
            else:
                print("=> no checkpoint found at '{}'".format(self.args.resume))

        # if self.args.evaluate:
        #     self.validate(val_loader, model, criterion)
        #     return
        metrics_dict = {
            'train_auc': [],
            'train_acc': [],
            'test_auc': [],
            'test_acc': []
        }
        for epoch in range(self.args.num_epoch):
            self.epoch = epoch
            # self.train0(train_loader, model, criterion, optimizer, epoch, self.args.num_epoch,self.args)
            # train for one epoch
            train_true, train_score,train_fusion,train_label,train_acc= self.train(train_loader, model, criterion, optimizer, epoch, self.args.num_epoch,self.args)
            # 计算AUC值
            train_auc = roc_auc_score(train_true, train_score)
            # evaluate on validation set
            test_true, test_score,test_fusion,test_label,test_acc = self.validate(val_loader, model, criterion, epoch, self.args.num_epoch,self.args)
            # evaluate on validation set
            # 计算AUC值
            test_auc = roc_auc_score(test_true, test_score)
            test1_true, test1_score,test1_fusion,test1_label,test1_acc = self.extract_validate(test1_loader, model, criterion, epoch, self.args.num_epoch,self.args)
            # 计算AUC值
            test1_auc = roc_auc_score(test1_true,test1_score)

            # if train_auc < 0.5:
            #     train_auc = 1 - train_auc
            # if test_auc < 0.5:
            #     test_auc = 1 - test_auc
            # if test1_auc < 0.5:
            #     test1_auc = 1 - test1_auc

            print(f"当前epoch{epoch},train AUC = {train_auc}, test AUC = {test_auc}, test1 AUC = {test1_auc}")

            # 构建字符串
            log_str = f"当前epoch {epoch}, train AUC = {train_auc:.4f}, train ACC = {train_acc:.4f}, test AUC = {test_auc:.4f}, test ACC = {test_acc:.4f}, test1 AUC = {test1_auc:.4f}, test1 ACC = {test1_acc:.4f}"

            # 保存到 txt 文件中（追加模式，不会覆盖之前的内容）
            log_path = os.path.join(self.results_dir, "training_log.txt")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_str + "\n")
            metrics_dict['train_auc'].append(train_auc)
            metrics_dict['train_acc'].append(train_acc)
            metrics_dict['test_auc'].append(test_auc)
            metrics_dict['test_acc'].append(test_acc)


            # if train_auc > 0.70  and test_auc > 0.65 and train_auc < 0.98 and test1_auc > 0.5:
            if train_auc > 0.70  and test_auc > 0.65:
            # if train_auc > 0.4:
                # is_best = test1_auc > self.best_score
                # if is_best:
                #     self.best_score = test1_auc
                #     self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    # 'best_score': self.best_score})
                    'best_score': test1_auc})
                scio.savemat(os.path.join(self.results_dir, 'all_feature_%s_%s_%s.mat'%(train_auc,test_auc,test1_auc)),
                            mdict={'Xtrain': np.array(train_fusion, dtype=np.float64),
                                    'Ytrain': np.mat(train_label, dtype=np.float64),
                                    'Xtest': np.mat(test_fusion, dtype=np.float64),
                                    'Ytest': np.mat(test_label, dtype=np.float64),
                                    'Xtest1': np.mat(test1_fusion, dtype=np.float64),
                                    'Ytest1': np.mat(test1_label, dtype=np.float64),
                                    # 'train_P': np.mat(train_P, dtype=np.float64),
                                    # 'train_P_hat': np.mat(train_P_hat, dtype=np.float64),
                                    # 'train_R': np.mat(train_R, dtype=np.float64),
                                    # 'train_R_hat': np.mat(train_R_hat, dtype=np.float64),
                                    # 'test_P': np.mat(test_P, dtype=np.float64),
                                    # 'test_P_hat': np.mat(test_P_hat, dtype=np.float64),
                                    # 'test_R': np.mat(test_R, dtype=np.float64),
                                    # 'test_R_hat': np.mat(test_R_hat, dtype=np.float64),
                                    # 'test1_P': np.mat(test1_P, dtype=np.float64),
                                    # 'test1_P_hat': np.mat(test1_P_hat, dtype=np.float64),
                                    # 'test1_R': np.mat(test1_R, dtype=np.float64),
                                    # 'test1_R_hat': np.mat(test1_R_hat, dtype=np.float64)
                                    # 'test1_R_hat': np.mat(test1_R_hat.detach().cpu().numpy(), dtype=np.float64),
                                    })
                                
                print(f"当前epoch{epoch},train AUC = {train_auc}, test AUC = {test_auc}, test1 AUC = {test1_auc} 参数已保存")
            # print(' ***train auc = {:.4f} best test auc={:.4f} at epoch {}'.format(train_auc,self.best_score, self.best_epoch))
            if scheduler is not None:
                scheduler.step()
            print('>')
        plot_metrics(metrics_dict, title='Metrics History', x_label='Epoch', y_label='Value', output_path=Path(os.path.join(self.results_dir, 'training_metrics.png')))
        return self.best_score, self.best_epoch

    
    
    def train(self, data_loader, model, criterion, optimizer, epoch, n_epoch,args):

        losses = AverageMeter()
        error = AverageMeter()
        acc = AverageMeter()
        train_sample_num = len (data_loader)
        # 记录训练集标签和融合特征
        # fusion_all = np.zeros((args.train_sample_num, 256), dtype='float32')
        fusion_all = np.zeros((train_sample_num, 1024), dtype='float32')
        # P_all = np.zeros((args.train_sample_num, 512), dtype='float32')
        # P_hat_all = np.zeros((args.train_sample_num, 512), dtype='float32')
        # R_all = np.zeros((args.train_sample_num, 512), dtype='float32')
        # R_hat_all = np.zeros((args.train_sample_num, 512), dtype='float32')
        label_all = np.array([])
        train_score = np.array([])
        train_true = np.array([])
        model.train()
        # for batch_idx, (CT_feature, Pathology_feature,CT_feature3, label,data_path) in enumerate(data_loader):
        for batch_idx, (CT_feature, Pathology_feature, label, data_path) in enumerate(data_loader):
            #在数据读取的时候获取一下患者特征的路径，将每个患者的融合特征重新保存一遍！！！
            if torch.cuda.is_available():
                CT_feature = CT_feature.cuda()
                # print(CT_feature.shape)
                Pathology_feature = Pathology_feature.cuda()
                # CT_feature3 = CT_feature3.cuda()
                label = label.type(torch.LongTensor).cuda()
                # label_1 = label.squeeze(0).long()
                # c = c.type(torch.FloatTensor).cuda()
                label_all = np.append(label_all,label.cpu())
            # pre, fusion= model(CT_feature=CT_feature, Pathology_feature=Pathology_feature,CT_feature3=CT_feature3)
                # pre, fusion, out_dict,_ = model(x_path = Pathology_feature, x_radio = CT_feature, phase = 'train', label = label)
                # pre, fusion, P, R, indiv_components,out_dict,proxy_loss= model(x_path = Pathology_feature, x_radio = CT_feature, phase = 'train', label = label)
                pre, fusion, P, R, indiv_components,out_dict = model(x_path = Pathology_feature, x_radio = CT_feature, phase = 'train', label = label)
                dataset = 'train_data'
            # loss, loss_dict = criterion(out_dict, label)
            # actual_batch_size = len(fusion)
            # fusion_all[batch_idx*actual_batch_size:(batch_idx+1)*actual_batch_size,:] = fusion.detach().cpu().numpy()
        
            
                loss_ = criterion[0](pre, label)
                # loss_knowledge = criterion[1](P.detach(), indiv_components[0]) + criterion[1](R.detach(), indiv_components[1]) + criterion[1](P.detach(), indiv_components[2]) + criterion[1](R.detach(), indiv_components[2]) + (torch.tensor(2, device=P.device) - criterion[1](P.detach(), indiv_components[3]) - criterion[1](R.detach(), indiv_components[3]) - criterion[1](P.detach(), indiv_components[1]) - criterion[1](R.detach(), indiv_components[0]))
                loss_cohort = criterion[1](out_dict,label)
                # loss_patience = criterion[2](indiv_components[0],label,dataset) + criterion[3](indiv_components[1],label,dataset) + criterion[4](indiv_components[2],label,dataset) + criterion[5](indiv_components[3],label,dataset)
                
                # loss = loss_ + self.args.alpha*loss_cohort + self.args.beta*proxy_loss
                loss = loss_ + self.args.alpha*loss_cohort
                # loss = loss_ + self.args.alpha * loss_knowledge + self.args.beta * loss_patience


                # loss = criterion[0](pre, label)
                # 记录错误率、正确率意见损失函数
                _, pred = pre.data.topk(1, dim=1)
                # print(pred)
                batch_size = label.size(0)
                error.update(torch.ne(pred.cpu().squeeze(), label.cpu()).float().sum().item() / batch_size, batch_size)
                acc.update(torch.eq(pred.cpu().squeeze(), label.cpu()).float().sum().item() / batch_size, batch_size)
                losses.update(loss.item(), batch_size)

                # P = P.cpu()
                # P_hat = P_hat.cpu()
                # R = R.cpu()
                # R_hat = R_hat.cpu()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_idx % 50 == 0:
                    res = '\t'.join([
                        'Epoch: [%d/%d]' % (epoch + 1, n_epoch),
                        'Iter: [%d/%d]' % (batch_idx + 1, len(data_loader)),
                        'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                        'Error %.4f (%.4f)' % (error.val * 100, error.avg * 100),
                        'Acc %.4f (%.4f)' % (acc.val * 100, acc.avg * 100),
                    ])
                    print(res)


                p = pre[:,1]
                fusion_all[batch_idx] = fusion.cpu().detach().numpy()
                train_score = np.append(train_score, p.cpu().detach().numpy())
                train_true = np.append(train_true, label.cpu())

            torch.save({
                'path_know_memory': model.path_know_memory.cpu(),
                'radio_know_memory': model.radio_know_memory.cpu(),
                'patient_bank': [pb.cpu() for pb in model.patient_bank]
            }, 'cmta_memory.pth')

    
        

        return train_true, train_score, fusion_all, label_all, acc.avg
    
    def validate(self, data_loader, model, criterion, epoch, n_epoch,args):
            losses = AverageMeter()
            error = AverageMeter()
            acc = AverageMeter()

            test_sample_num = len (data_loader)

            fusion_all = np.zeros((test_sample_num, 1024), dtype='float32')
            P_all = np.zeros((args.test_sample_num, 512), dtype='float32')
            P_hat_all = np.zeros((args.test_sample_num, 512), dtype='float32')
            R_all = np.zeros((args.test_sample_num, 512), dtype='float32')
            R_hat_all = np.zeros((args.test_sample_num, 512), dtype='float32')
            label_all = np.array([])
            test_score = np.array([])
            test_true = np.array([])

            model.eval()


                # for batch_idx, (CT_feature, Pathology_feature,CT_feature3, label,data_path) in enumerate(data_loader):
            for batch_idx, (CT_feature, Pathology_feature, label,data_path) in enumerate(data_loader):
                if torch.cuda.is_available():
                    CT_feature = CT_feature.cuda()
                    Pathology_feature = Pathology_feature.cuda()
                        # CT_feature3 = CT_feature3.cuda()

                    label = label.type(torch.LongTensor).cuda()
                    label_all = np.append(label_all,label.cpu())

                    with torch.no_grad():
                        # 加载 memory 和 patient_bank

                        # pre, fusion= model(CT_feature=CT_feature, Pathology_feature=Pathology_feature,CT_feature3=CT_feature3)
                        # pre, fusion, P, R, indiv_components,out_dict,proxy_loss = model(x_path = Pathology_feature, x_radio = CT_feature, phase = 'test', label = label)
                        pre, fusion, P, R, indiv_components,out_dict = model(x_path = Pathology_feature, x_radio = CT_feature, phase = 'test', label = label)
                        dataset = 'test'

                        # loss, loss_dict = criterion(out_dict, label)
                        # actual_batch_size = len(fusion)
                        # fusion_all[batch_idx*actual_batch_size:(batch_idx+1)*actual_batch_size,:] = fusion.detach().cpu().numpy()
                        loss_ = criterion[0](pre, label)
                        # loss_knowledge = criterion[1](P.detach(), indiv_components[0]) + criterion[1](R.detach(), indiv_components[1]) + criterion[1](P.detach(), indiv_components[2]) + criterion[1](R.detach(), indiv_components[2]) + (torch.tensor(2, device=P.device) - criterion[1](P.detach(), indiv_components[3]) - criterion[1](R.detach(), indiv_components[3]) - criterion[1](P.detach(), indiv_components[1]) - criterion[1](R.detach(), indiv_components[0]))
                        loss_cohort = criterion[1](out_dict,label)
                        # loss_patience = criterion[2](indiv_components[0],label,dataset) + criterion[3](indiv_components[1],label,dataset) + criterion[4](indiv_components[2],label,dataset) + criterion[5](indiv_components[3],label,dataset)
                        
                        # loss = loss_ + self.args.alpha*loss_cohort + args.beta*proxy_loss
                        loss = loss_ + self.args.alpha*loss_cohort
                        # loss = loss_ + self.args.alpha * loss_knowledge + 0.3* loss_patience
                        # loss = criterion[0](pre, label)
                        # 记录错误率、正确率意见损失函数
                        _, pred = pre.data.topk(1, dim=1)
                        # print(pred)
                        batch_size = label.size(0)
                        error.update(torch.ne(pred.cpu().squeeze(), label.cpu()).float().sum().item() / batch_size, batch_size)
                        acc.update(torch.eq(pred.cpu().squeeze(), label.cpu()).float().sum().item() / batch_size, batch_size)
                        losses.update(loss.item(), batch_size)
                        # 记录预测标签与真实标签
                        p = pre[:,1]

                        # test_score = np.append(test_score, p.cpu())
                        # test_true = np.append(test_true, label.cpu())

                        # P = P.cpu()
                        # P_hat = P_hat.cpu()
                        # R = R.cpu()
                        # R_hat = R_hat.cpu()

                if batch_idx % 1 == 0:
                    res = '\t'.join([
                        # 'Epoch: [%d/%d]' % (epoch + 1, n_epoch),
                        'Iter: [%d/%d]' % (batch_idx + 1, len(data_loader)),
                        'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                        'Error %.4f (%.4f)' % (error.val * 100, error.avg * 100),
                        'Acc %.4f (%.4f)' % (acc.val * 100, acc.avg * 100),
                    ])
                print(res)
                p = pre[:,1]
                fusion_all[batch_idx] = fusion.cpu().detach().numpy()
                test_score = np.append(test_score, p.cpu().detach().numpy())
                test_true = np.append(test_true, label.cpu())
            return test_true, test_score, fusion_all, label_all, acc.avg


    def extract_validate(self, data_loader, model, criterion, epoch, n_epoch,args):
            losses = AverageMeter()
            error = AverageMeter()
            acc = AverageMeter()

            test1_sample_num = len (data_loader)
            
            fusion_all = np.zeros((test1_sample_num, 1024), dtype='float32')
            P_all = np.zeros((args.test1_sample_num, 512), dtype='float32')
            P_hat_all = np.zeros((args.test1_sample_num, 512), dtype='float32')
            R_all = np.zeros((args.test1_sample_num, 512), dtype='float32')
            R_hat_all = np.zeros((args.test1_sample_num, 512), dtype='float32')
            label_all = np.array([])
            test1_score = np.array([])
            test1_true = np.array([])

            model.eval()
            # for batch_idx, (CT_feature, Pathology_feature,CT_feature3, label,data_path) in enumerate(data_loader):
            for batch_idx, (CT_feature, Pathology_feature, label, data_path) in enumerate(data_loader):
                if torch.cuda.is_available():
                    CT_feature = CT_feature.cuda()
                    Pathology_feature = Pathology_feature.cuda()
                    # CT_feature3 = CT_feature3.cuda()

                    label = label.type(torch.LongTensor).cuda()
                    label_all = np.append(label_all,label.cpu())

                with torch.no_grad():
                    # pre, fusion= model(CT_feature=CT_feature, Pathology_feature=Pathology_feature,CT_feature3=CT_feature3)
                    # pre, fusion, P, P_hat, R, R_hat= model(CT_feature=CT_feature, Pathology_feature=Pathology_feature)
                    # actual_batch_size = len(fusion)
                    # fusion_all[batch_idx*actual_batch_size:(batch_idx+1)*actual_batch_size,:] = fusion.detach().cpu().numpy()
                    # pre, fusion, P, R, indiv_components,out_dict,proxy_loss = model(x_path = Pathology_feature, x_radio = CT_feature, phase = 'test1', label = label)
                    pre, fusion, P, R, indiv_components,out_dict = model(x_path = Pathology_feature, x_radio = CT_feature, phase = 'test1', label = label)
                    dataset = 'test1'

                    # loss, loss_dict = criterion(out_dict, label)
                    # actual_batch_size = len(fusion)
                    # fusion_all[batch_idx*actual_batch_size:(batch_idx+1)*actual_batch_size,:] = fusion.detach().cpu().numpy()
                    loss_ = criterion[0](pre, label)
                    # loss_knowledge = criterion[1](P.detach(), indiv_components[0]) + criterion[1](R.detach(), indiv_components[1]) + criterion[1](P.detach(), indiv_components[2]) + criterion[1](R.detach(), indiv_components[2]) + (torch.tensor(2, device=P.device) - criterion[1](P.detach(), indiv_components[3]) - criterion[1](R.detach(), indiv_components[3]) - criterion[1](P.detach(), indiv_components[1]) - criterion[1](R.detach(), indiv_components[0]))
                    loss_cohort = criterion[1](out_dict,label)
                    # loss_patience = criterion[2](indiv_components[0],label,dataset) + criterion[3](indiv_components[1],label,dataset) + criterion[4](indiv_components[2],label,dataset) + criterion[5](indiv_components[3],label,dataset)
                    
                    # loss = loss_ + self.args.alpha*loss_cohort + args.beta*proxy_loss
                    loss = loss_ + self.args.alpha*loss_cohort 
                    # loss = loss_ + self.args.alpha * loss_knowledge + 0.3* loss_patience
                    # loss = criterion[0](pre, label)
                    # 记录错误率、正确率意见损失函数
                    _, pred = pre.data.topk(1, dim=1)
                    # print(pred)
                    batch_size = label.size(0)
                    error.update(torch.ne(pred.cpu().squeeze(), label.cpu()).float().sum().item() / batch_size, batch_size)
                    acc.update(torch.eq(pred.cpu().squeeze(), label.cpu()).float().sum().item() / batch_size, batch_size)
                    losses.update(loss.item(), batch_size)
                    # 记录预测标签与真实标签
                    p = pre[:,1]
                    # test1_score = np.append(test1_score, p.cpu())
                    # test1_true = np.append(test1_true, label.cpu())

                    # P = P.cpu()
                    # P_hat = P_hat.cpu()
                    # R = R.cpu()
                    # R_hat = R_hat.cpu()

                if batch_idx % 50 == 0:
                    res = '\t'.join([
                        # 'Epoch: [%d/%d]' % (epoch + 1, n_epoch),
                        'Iter: [%d/%d]' % (batch_idx + 1, len(data_loader)),
                        'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                        'Error %.4f (%.4f)' % (error.val * 100, error.avg * 100),
                        'Acc %.4f (%.4f)' % (acc.val * 100, acc.avg * 100),
                    ])
                    print(res)
                    fusion_all[batch_idx] = fusion.cpu().detach().numpy()
                    test1_score = np.append(test1_score, p.cpu().detach().numpy())
                    test1_true = np.append(test1_true, label.cpu())
            return test1_true, test1_score, fusion_all, label_all, acc.avg


    def save_checkpoint(self, state):
        # if self.filename_best is not None:
            # os.remove(self.filename_best)
        self.filename_best = os.path.join(self.results_dir,
                                          'model_best_{score:.4f}_{epoch}.pth'.format(score=state['best_score'],
                                                                                          epoch=state['epoch']))
        print('save best model {filename}'.format(filename=self.filename_best))
        torch.save(state, self.filename_best)
