# 导入Trainer类，用于训练时序预测模型
from trainers.tsf_trainer import Trainer
# 导入numpy库，用于数值计算
import numpy as np
# 导入random库，用于生成随机数
import random
# 导入yaml库，用于解析YAML配置文件
import yaml
# 导入warnings库，用于处理警告信息
import warnings
# 导入torch库，用于深度学习模型的构建和训练
import torch
from datetime import datetime
import wandb
# 设置随机种子，确保实验可重复性
seed = 1234


if __name__ == '__main__':
    # 忽略FutureWarning类型的警告
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 设置PyTorch随机种子
    torch.manual_seed(seed)
    # 设置NumPy随机种子
    np.random.seed(seed)
    # 设置Python随机种子
    random.seed(seed)

    # 检查可用设备并输出信息
    if torch.cuda.is_available():
        print('nvidia CUDA')
        print(torch.cuda.current_device())
    elif torch.mps.is_available():
        print('Apple MPS')
    else:
        print('CPU')

    run = wandb.init(
        #设置项目名称
        project="sampler test",
        #设计runs记录的名称（这里用时间）
        name = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        #设置runs记录的描述
        config={
            "cycle": "False",
        },
    )        


    # 创建Trainer实例，传入基础配置文件路径
    trainer = Trainer('configs/bases/base.yaml')
    # 开始训练模型
    trainer.train()
