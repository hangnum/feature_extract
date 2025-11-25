import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as scio
import torch
import torchvision.transforms as transforms
from PIL import Image
from pandas.core.frame import DataFrame
from torch.utils.data import DataLoader, Dataset


# 获取当前脚本的文件路径
script_dir = Path(__file__).resolve().parent

# 将项目的根目录添加到sys.path
project_root = script_dir.parents[2]
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

# 确保脚本的工作目录为项目的根目录
os.chdir(project_root)
# mean = [0.49139968, 0.48215841, 0.44653091]
# stdv = [0.24703223, 0.24348513, 0.26158784]
transformTransfer = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=stdv),
])


# 构造数据集
class MyDataset(Dataset):
    def __init__(self, CT_data_file,P_data_file, dataset, transform=True):
        # self.CT_feature_path = os.path.join(data_path, 'AP_mRMR_Feature.mat')
        self.CT_data_file = CT_data_file
        self.transform = transform
        self.dataset = dataset
        self.CT_size = 0
        self.CT_list = []

        self.Pathology_data_file = P_data_file
        self.Pathology_size = 0
        self.Pathology_list = []

        # 获取当前文件所在目录的绝对路径
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # 得到 train1_log.txt 文件的绝对路径
        CT_txt_path = os.path.abspath(os.path.join(current_directory, self.CT_data_file))
        Pathology_txt_path = os.path.abspath(os.path.join(current_directory, self.Pathology_data_file))

        if not os.path.isfile(CT_txt_path):
            print(CT_txt_path + 'does not exist!!')
        CT_file = open(CT_txt_path)  # 读取包含CT特征文本信息

        if not os.path.isfile(Pathology_txt_path):
            print(Pathology_txt_path + 'does not exist!!')
        Pathology_file = open(Pathology_txt_path)  # 读取包含病例特征文本信息



        for f in CT_file:
            self.CT_list.append(f)
            self.CT_size += 1
        for p in Pathology_file:
            self.Pathology_list.append(p)
        # CT_feature_data = scio.loadmat(self.CT_feature_path)  # 包含了训练集、测试集的标签和特征

        # if dataset == 'train':
        #     clinical_path = os.path.join(data_path, dataset+'.csv')
        #     self.clinical_data = pd.read_csv(clinical_path, encoding="gb2312",engine='python')
        #     self.ct_feature = CT_feature_data['Xtrain1']
        #     for f in self.ct_feature:
        #         self.feature_list.append(f)
        #         self.size += 1
        #     self.label = CT_feature_data['Ytrain1']
        # elif dataset == 'test':
        #     clinical_path = os.path.join(data_path, dataset+'.csv')
        #     self.clinical_data = pd.read_csv(clinical_path, encoding="UTF-8",engine='python')
        #     self.ct_feature = CT_feature_data['Xtest1']
        #     for f in self.ct_feature:
        #         self.feature_list.append(f)
        #         self.size += 1
        #     self.label = CT_feature_data['Ytest1']
    def __len__(self):
        return self.CT_size

    def __getitem__(self, idx):
        CT_feature_path = self.CT_list[idx].split('*')[0]
        if not os.path.isfile(CT_feature_path):
            print(CT_feature_path + ' ' + 'does not exist!')
            return None
        CT_feature = scio.loadmat(CT_feature_path)['feature_map']

        Pathology_feature_path = self.Pathology_list[idx].split('*')[0]
        if not os.path.isfile(Pathology_feature_path):
            print(Pathology_feature_path + ' ' + 'does not exist!')
            return None
        Pathology_feature = scio.loadmat(Pathology_feature_path)['feature_map']


        label = int(self.CT_list[idx].split('*')[2])


        # CT_feature = self.ct_feature[idx]
        # Clinical_feature = self.clinical_data.iloc[:, 1:]  # .iloc用于指定DataFrem中指定的行列
        # Clinical_feature = Clinical_feature.iloc[idx,:]
        # Clinical_feature = torch.Tensor(Clinical_feature.values)
        # label = self.label[idx]
        # image = Image.open(image_path)
        # label = int(self.names_list[idx].split('*')[2])
        # PatienceName = os.path.dirname(image_path)

        # if self.transform is not None:
        #     image = transformTransfer(image)
        # return image, label, PatienceName
        return CT_feature, Pathology_feature, label

# 更改后的新数据集接口
def get_readData(args):
    train_dataset = MyDataset(CT_data_file=os.path.join(args.txt_dir,r'train1_log.txt'), P_data_file=os.path.join(args.txt_dir,r'train_BL_log.txt'),dataset = 'train', transform=True)
    # train_dataset = MyDataset(CT_data_file=os.path.join(args.txt_dir,'train_PCP_log.txt'), P_data_file=os.path.join(args.txt_dir,'train_Pathology_epoch_10_log.txt'),dataset = 'train', transform=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    test_dataset = MyDataset(CT_data_file=os.path.join(args.txt_dir,r'test1_log.txt'), P_data_file=os.path.join(args.txt_dir,r'test_BL_log.txt'),dataset = 'test', transform=True)
    # test_dataset = MyDataset(CT_data_file=os.path.join(args.txt_dir,'test_PCP_log.txt'), P_data_file=os.path.join(args.txt_dir,'test_Pathology_epoch_10_log.txt'),dataset = 'test', transform=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    test1_dataset = NEWDataset(CT_data_file=os.path.join(args.txt_dir,r'test1_log.txt'), Pathology_data_file=os.path.join(args.txt_dir,r'test1_BL_log.txt'),dataset = 'test1', transform=True)
    test1_loader = DataLoader(test1_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader, test1_loader










class NEWDataset(Dataset):
    # def __init__(self, CT_data_file,Pathology_data_file,CT_data_file3, dataset, transform=True):
    def __init__(self, CT_data_file,Pathology_data_file, dataset, transform=True):
        # self.CT_feature_path = os.path.join(data_path, 'AP_mRMR_Feature.mat')
        self.CT_data_file = CT_data_file
        self.Pathology_data_file = Pathology_data_file
        # self.CT_data_file3 = CT_data_file3
        self.transform = transform
        self.dataset = dataset
        self.CT_size = 0
        self.CT_list = []
        self.Pathology_list = []
        # self.CT_list3 = []

        # # 获取当前文件所在目录的绝对路径
        current_directory = os.path.dirname(os.path.abspath(__file__))
        # current_directory = os.path.abspath(os.path.join(current_directory, ".."))
        # 得到 train1_log.txt 文件的绝对路径
        CT_txt_path = os.path.abspath(os.path.join(current_directory, self.CT_data_file))
        Pathology_txt_path = os.path.abspath(os.path.join(current_directory, self.Pathology_data_file))
        # CT_txt_path3 = os.path.abspath(os.path.join(current_directory, self.CT_data_file3))

        if not os.path.isfile(CT_txt_path):
            print(CT_txt_path + 'does not exist!!')
        CT_file = open(CT_txt_path)  # 读取包含CT特征文本信息

        if not os.path.isfile(Pathology_txt_path):
            print(Pathology_txt_path + 'does not exist!!')
        Pathology_file = open(Pathology_txt_path)  # 读取包含病例特征文本信息

        # if not os.path.isfile(CT_txt_path3):
        #     print(CT_txt_path3 + 'does not exist!!')
        # CT_file3 = open(CT_txt_path3)  # 读取包含病例特征文本信息



        for f in CT_file:
            self.CT_list.append(f)
            self.CT_size += 1
        for f in Pathology_file:
            self.Pathology_list.append(f)
        # for f in CT_file3:
        #     self.CT_list3.append(f)

    def __len__(self):
        return self.CT_size

    def __getitem__(self, idx):
        CT_feature_path = self.CT_list[idx].split('*')[0]
        if not os.path.isfile(CT_feature_path):
            print(CT_feature_path + ' ' + 'does not exist!')
            return None
        CT_feature = scio.loadmat(CT_feature_path)['feature_map']

        data_path = CT_feature_path
        Pathology_feature_path = self.Pathology_list[idx].split('*')[0]
        if not os.path.isfile(Pathology_feature_path):
            print(Pathology_feature_path + ' ' + 'does not exist!')
            return None
        Pathology_feature = scio.loadmat(Pathology_feature_path)['feature_map']

        # CT_feature_path3 = self.CT_list[idx].split('*')[0]
        # if not os.path.isfile(CT_feature_path3):
        #     print(CT_feature_path3 + ' ' + 'does not exist!')
        #     return None
        # CT_feature3 = scio.loadmat(CT_feature_path3)['feature_map']

        # max_length = max(len(CT_feature),len(Pathology_feature),len(CT_feature3))
        # max_length = max(len(CT_feature),len(Pathology_feature))
        # feature1_zeros_rows = np.zeros((max_length-len(Pathology_feature), 3904))
        # feature2_zeros_rows = np.zeros((max_length-len(CT_feature), 3904))
        # # feature3_zeros_rows = np.zeros((max_length-len(CT_feature3), 3904))

        # CT_feature = np.vstack([CT_feature,feature1_zeros_rows])
        # Pathology_feature = np.vstack([Pathology_feature,feature2_zeros_rows])
        # CT_feature3 = np.vstack([CT_feature,feature3_zeros_rows])


        label = int(self.CT_list[idx].split('*')[2])
        # return CT_feature, Pathology_feature,CT_feature3, label,data_path
        return CT_feature, Pathology_feature, label, data_path




def get_MultiData(args):
    # train_dataset = NEWDataset(CT_data_file=os.path.join(args.txt_dir,'train_NP_log.txt'), Pathology_data_file=os.path.join(args.txt_dir,'train_PCP_log.txt'),CT_data_file3=os.path.join(args.txt_dir,'train_CMP_log.txt'),dataset = 'train', transform=True)
    train_dataset = NEWDataset(CT_data_file=os.path.join(args.txt_dir,r'train_A.txt'), Pathology_data_file=os.path.join(args.txt_dir,r'train_P.txt'),dataset = 'train', transform=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    test_dataset = NEWDataset(CT_data_file=os.path.join(args.txt_dir,r'val_A.txt'), Pathology_data_file=os.path.join(args.txt_dir,r'val_P.txt'),dataset = 'test', transform=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    test1_dataset = NEWDataset(CT_data_file=os.path.join(args.txt_dir,r'val_A.txt'), Pathology_data_file=os.path.join(args.txt_dir,r'val_P.txt'),dataset = 'test1', transform=True)
    test1_loader = DataLoader(test1_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, test1_loader


create_dataloaders = get_MultiData
