import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="Configurations for Survival Analysis on TCGA Data.")
    parser.add_argument(
        "--CT_data_root_dir", type=str, default="/mnt/yxyxlab/xjh/second/划分数据集_筛选后/TXT_test1/jiangmen_CT_CMTA",help="Data directory to  CT features ")
    parser.add_argument(
        "--Pathology_data_root_dir", type=str, default="/mnt/yxyxlab/xjh/second/划分数据集_筛选后/TXT_test1/jiangmen_BL_CMTA",help="Data directory to  Pathology features ")
    parser.add_argument('--txt_dir', type=str, default='/home/wwt/data/outputs/experiments_output/txt', help='TXT Path for saving the text')
    parser.add_argument('--category', type=list, default=['0', '1'], help='Two types of data labels')
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducible experiment (default: 1)")   # 原 1
    parser.add_argument(
        "--which_splits", type=str, default="5foldcv", help="Which splits folder to use in ./splits/ (Default: "
                                                            "./splits/5foldcv "
    )
    parser.add_argument("--resume_Select",  default=r'/home/yxyxlab_public/xjh/CMTA/CMTA/interpretability_new/pth/model_best_0.6413_378.pth')
    parser.add_argument(
        "--dataset",
        type=str,
        default="SGD",
        help='Which cancer type within ./splits/<which_dataset> to use for training. Used synonymously for "task" ('
             'Default: tcga_blca_100)',
    )
    parser.add_argument("--log_data", action="store_true", default=True, help="Log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="Evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH",
                        help="Path to latest checkpoint (default: none)")

    # Model Parameters.
    parser.add_argument(
        "--model",
        type=str,
        default="cmta",
        help="Type of model (Default: mcat)",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=[
            "small",
            "large",
        ],
        default="large",
        help="Size of some models (Transformer)",
    )
    parser.add_argument(
        "--modal",
        type=str,
        choices=["omic", "path", "pathomic", "cluster", "coattn"],  # 原 coattn
        default="coattn",
        help="Specifies which modalities to use / collate function in dataloader.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        choices=["concat", "bilinear"],  # 原 concat
        default="concat",
        help="Modality fuison strategy",
    )
    parser.add_argument("--alpha", type=float, default=0.5, help="hyper-parameter of loss function")
    parser.add_argument("--beta", type=float, default=0.1, help="hyper-parameter of loss function")
    # parser.add_argument("--num_cluster", type=int, default=64, help="hyper-parameter of loss function")
    # parser.add_argument("--bank_length", type=int, default=1000, help="hyper-parameter of loss function")
    parser.add_argument("--update_rat", type=float, default=0.1, help="hyper-parameter of loss function")


    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam",
                                                          "AdamW", "RAdam", "PlainRAdam", "Lookahead"],
                        default="AdamW")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine", "cosine_rest"],
                        default="cosine")
    parser.add_argument("--num_epoch", type=int, default=50, help="Maximum number of epochs to train (default: 20)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch Size (Default: 1, due to varying bag sizes)")
    parser.add_argument("--train_sample_num", type=int, default=124, help="训练集样本数量")
    parser.add_argument("--test_sample_num", type=int, default=52, help="测试集样本数量")
    parser.add_argument("--test1_sample_num", type=int, default=83, help="测试集样本数量")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--loss", type=str, default="CohortLoss", help="slide-level classification loss function (default: ce)")
    parser.add_argument("--weighted_sample", action="store_true", default=True, help="Enable weighted sampling")
    args = parser.parse_args()
    return args
