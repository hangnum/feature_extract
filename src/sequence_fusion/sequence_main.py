import os
import sys
import csv
import time
import numpy as np
import torch.nn as nn


os.environ['CUDA_VISIBLE_DEVICES']= "0"

# 获取当前脚本的文件路径




script_dir = os.path.dirname(os.path.abspath(__file__))

# 将项目的根目录添加到sys.path
project_root = os.path.abspath(os.path.join(script_dir, "..",'..'))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "datasets"))
# sys.path.append(os.path.join(project_root,"datasets"))
# 确保脚本的工作目录为项目的根目录  
os.chdir(project_root)
print(f"Project Root: {project_root}")
print(f"sys.path: {sys.path}")

from datasets.clincial_dataset import get_MultiData
from datasets.dataset_survival import Generic_MIL_Survival_Dataset
from utils.options import parse_args
from utils.util import get_split_loader, set_seed

from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler


def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    print("当前使用CT数据为PCP")
    # create results directory
    results_dir = "/home/wwt/data/outputs/experiments_output/large_resnet18_fusion_results/{opt}-[{fusion}]-[{alpha}]--[{wg}]-[{lr}]-[{time}]".format(
        # dataset=args.dataset,
        opt = args.optimizer,
        fusion=args.fusion,
        alpha=args.alpha,
        wg = args.weight_decay,
        lr = args.lr,

        time=time.strftime("%Y-%m-%d]-[%H-%M-%S"),
    )
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 5-fold cross validation
    header = ["folds", "fold 0", "fold 1", "fold 2", "fold 3", "fold 4", "mean", "std"]
    best_epoch = ["best epoch"]
    best_score = ["best cindex"]

    # Data loaders 
    # train_loader, test_loader = get_readData(args)
    train_loader, test_loader, test1_loader = get_MultiData(args)

    # build model, criterion, optimizer, schedular
    if args.model == "cmta":
        from models.sequence_fusion.sequence_network import CMTA
        from models.sequence_fusion.sequence_engine import Engine

        # print(train_dataset.omic_sizes)
        # print(train_loader.dataset.clinical_data.shape[1]-1)
        model_dict = {
            # "omic_sizes": train_dataset.omic_sizes, # 训练集中genomic_features类别与6种标志物中的类别交集的个数
            # "clinical_sizes":train_loader.dataset.clinical_data.shape[1]-1, # 临床数据指标个数
            "n_classes": 2,
            "fusion": args.fusion,
            "model_size": args.model_size,
        }
        model = CMTA(**model_dict)
        criterion = define_loss(args)   
        # criterion = nn.CrossEntropyLoss()
        optimizer = define_optimizer(args, model)
        scheduler = define_scheduler(args, optimizer)
        

        engine = Engine(args, results_dir, 0)
    else:
        raise NotImplementedError(
            "Model [{}] is not implemented".format(args.model)
        )
    # start training
    score, epoch = engine.learning(
        model, train_loader, test_loader,test1_loader, criterion, optimizer, scheduler
    )
    
    # save best score and epoch for each fold
    best_epoch.append(epoch)
    best_score.append(score)

    # finish training
    # mean and std
    best_epoch.append("~")
    best_epoch.append("~")
    best_score.append(np.mean(best_score[1:6]))
    best_score.append(np.std(best_score[1:6]))

    csv_path = os.path.join(results_dir, "results.csv")
    print("############", csv_path)
    with open(csv_path, "w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        writer.writerow(best_epoch)
        writer.writerow(best_score)


if __name__ == "__main__":
    args = parse_args()
    # for i in ["SGD", "Adam","AdamW", "RAdam", "PlainRAdam", "Lookahead"]:
    #     args.optimizer = i
    #     for j in np.arange(0.1, 1.1, 0.1):
    #         args.alpha = j
    #         results = main(args)
    results = main(args)
    print("finished!")
