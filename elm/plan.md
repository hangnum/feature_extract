# 任务工作流程

- [x] 梳理原始 Python 和 MATLAB 流程、列出主要问题
- [x] 设计统一的数据协议（特征矩阵/标签格式、路径配置）并确定 Python 重构方案
- [x] 用 Python 实现特征聚合与数据集划分（生成 `feature_{data_type}_map.mat`）
- [x] 用 Python 实现归一化与 U 检验特征筛选（替代 `feature_normalized.mat` 和 `feature_Utest_Part1_VP.mat`）
- [x] 用 Python 实现 ELM 训练评估与模型筛选（替代 MATLAB 训练脚本）
- [x] 工程化拆分：JSON 配置文件 + CLI 参数（适配 conda Python 3.9 环境）
- [x] 提供最小运行示例：`run_synthetic_demo.py` 自动生成随机数据并跑通整条流水线
- [ ] 验证关键逻辑（形状/标签/归一化一致性），整理使用说明 / 补充更多测试
