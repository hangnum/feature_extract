# 计划

## 数据预处理

    1.1.数据读取
        利用rglob读取数据
        解析图片信息，将每个病人的每个模态下的图片路径列表以字典的形式保存
        例子：
            {
                病人id：{
                    hospital: ''
                    label: int # 原本是grade0,grade1，映射成0,1
                    image_path: {
                        A: []
                        P: [] # A, P分别代表模态名称，列表存储图片路径
                    }
                }
            }
    1.2.数据预处理
        按照医院（JM按照7:3训练集和内验, 其他医院全部为外验），把病人们划分训练集，内验集，测试集。不同的模态需要分开
    1.3.数据保存
        训练集，内验集，测试集划分文件train_{modality}.csv，val_{modality}.csv，test_{modality}.csv（格式为两列image_path,label），保存在项目的/data/文件夹下

## 迁移和微调模型(单个模态)

    2.1.加载模型
        按照配置文件或者命令行输入，加载所需要的模型名称（resnet18,resnet50，swin-transformer-t）等
        迁移时，按照普遍情况进行是否冻结：目前我的期望时resnet18不冻结，resnet50大部分冻结
    2.2.加载数据
        根据上一步的train_{modality}.csv，val_{modality}.csv，test_{modality}.csv文件加载数据集
    2.3.微调和训练
        根据数据微调和训练模型
        实现多种损失函数交叉熵，FocalLoss，AsymmetricLoss等
        要有断点重连和保留最佳参数最佳模型，记录训练效果
        保存在模型参数D:\outputs\feature_extract\model_pth\{model_name}下
        在.\best_hparams\下给每个模型维护一个yaml，记录表现最好的超参数

## 特征提取(单个模态)

    3.1.读取模型
        读取配置文件所需的特征提取模型。
        加载该模型库下保留的最好的参数
    3.2.特征提取
        读取配置对该模态的所有图片进行特征提取
        分别保存在.\data\processed_data\{modality}\{hosipital_name}\{label}\下

## 对于多个模态分别提取完成之后

    4.1.读取
        在各个模态分别微调完，提取完特征之后。读取特征文件的地址
    4.2.生成对应文件
        按照医院（JM按照7:3训练集和内验, 其他医院全部为外验），把病人们划分训练集，内验集，测试集。不同的模态需要分开
        将特征文件再次划分成feature_train_{modality}.csv，feature_val_{modality}.csv，feature_test_{modality}.csv
        注意，这里的病人在不同的模态的排序要保持对应，比如说：
            病人A，在feature_train_A.csv中排第十个，那么他在feature_train_P.csv也必须要排在第十个
