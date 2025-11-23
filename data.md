# 数据格式

我的数据格式：
    1.224*224*1的图片
    2.对应的标签['grade0','grade1']
    3.文件路径root_dir: 'D:\data\raw\Grade'
    4.样例：
        D:\data\raw\Grade\JM\fold1\grade0\202009344\A\202009344_a_slice_005.png
        D:\data\raw\Grade\JM\fold1\grade0\202027938\T1\202027938_t1_slice_027.png
    5.解释：
        JM:医院名称
        fold1:折
        grade0:标签
        202009344:病人的id
        A:图片模态
        202009344_a_slice_005.png:病人的图片
    6.数据集：
        JM为训练和内验数据集
        其他医院为测试数据集（外验数据集）
    7.只有JM下有fold1 fold2 fold3 fold4 fold5
    8.模态：
        虽然每个病人下有多个模态，但是本次实验只使用A和P模态
