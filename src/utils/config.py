"""
配置管理系统

支持从YAML文件加载配置，并可通过命令行参数覆盖
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class DataConfig:
    """数据相关配置"""
    root_dir: str = r"D:\data\raw\Grade"
    modalities: list = field(default_factory=lambda: ['A', 'P'])
    train_ratio: float = 0.7
    image_size: int = 224
    num_workers: int = 4


@dataclass
class ModelConfig:
    """模型相关配置"""
    name: str = "resnet18"  # resnet18, resnet50, swin_t
    pretrained: bool = True
    num_classes: int = 2
    freeze_stages: int = 0  # ResNet50时设为3


@dataclass
class TrainingConfig:
    """训练相关配置"""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    loss_type: str = "ce"  # ce, focal, asymmetric
    optimizer: str = "adam"  # adam, sgd, adamw
    scheduler: str = "cosine"  # cosine, step, plateau
    early_stop_patience: int = 10
    device: str = "cuda"
    

@dataclass
class AugmentationConfig:
    """数据增强配置"""
    horizontal_flip: bool = True
    rotation_degrees: int = 15
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = ""
    seed: int = 42
    output_dir: str = r"D:\outputs\feature_extract"
    save_best_only: bool = True
    log_interval: int = 10
    

@dataclass
class Config:
    """总配置类"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        从YAML文件加载配置
        
        Args:
            yaml_path: YAML配置文件路径
        
        Returns:
            配置对象
        """
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'augmentation' in config_dict:
            config.augmentation = AugmentationConfig(**config_dict['augmentation'])
        if 'experiment' in config_dict:
            config.experiment = ExperimentConfig(**config_dict['experiment'])
        
        return config
    
    def to_yaml(self, yaml_path: str) -> None:
        """
        保存配置到YAML文件
        
        Args:
            yaml_path: 保存路径
        """
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'augmentation': asdict(self.augmentation),
            'experiment': asdict(self.experiment)
        }
        
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            配置字典
        """
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'augmentation': asdict(self.augmentation),
            'experiment': asdict(self.experiment)
        }
    
    def update_from_args(self, args: Any) -> None:
        """
        从命令行参数更新配置
        
        Args:
            args: argparse解析的参数
        """
        # 更新实验名称
        if hasattr(args, 'modality'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment.name = f"{self.model.name}_{args.modality}_{timestamp}"
        
        # 更新其他参数
        for key, value in vars(args).items():
            if value is not None:
                # 尝试更新各个配置部分
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
                elif hasattr(self.model, key):
                    setattr(self.model, key, value)
                elif hasattr(self.training, key):
                    setattr(self.training, key, value)
                elif hasattr(self.augmentation, key):
                    setattr(self.augmentation, key, value)
                elif hasattr(self.experiment, key):
                    setattr(self.experiment, key, value)


def create_default_config() -> Config:
    """
    创建默认配置
    
    Returns:
        默认配置对象
    """
    return Config()


def save_default_config(output_path: str) -> None:
    """
    保存默认配置模板
    
    Args:
        output_path: 输出路径
    """
    config = create_default_config()
    config.to_yaml(output_path)
    print(f"默认配置已保存至: {output_path}")
