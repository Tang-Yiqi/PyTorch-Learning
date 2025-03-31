import torch
from torch.optim.optimizer import Optimizer

# 定义 SGD 类，继承自 Optimizer，表示随机梯度下降优化器
class MySGD(Optimizer):
    # 初始化函数，设置优化器的超参数和需要更新的参数
    def __init__(self, params, lr=0.01):
        # 检查学习率是否合法（不能为负数）
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        # 将所有超参数放到一个字典中，便于后续使用
        defaults = dict(lr=lr)
        
        # 调用父类 Optimizer 的构造函数，传入参数组和默认超参数
        super().__init__(params, defaults)#父类Optimizer会将params, defaults传入self的param_groups中

    # 使用 @torch.no_grad() 装饰器避免在更新时计算梯度
    @torch.no_grad()
    def step(self):
        # 遍历所有参数组（一个优化器可以管理多个参数组）
        for group in self.param_groups:
            # 遍历当前参数组中的所有参数
            for p in group['params']:
                if p.grad is None:# 如果参数没有梯度，则跳过更新
                    continue
                else:
                    GradP = p.grad  # 获取参数的梯度(∂loss / ∂p)

                    # 根据更新公式，使用梯度下降：p = p - lr * GradP
                    p.add_(GradP, alpha=-group['lr'])#注意这里add_是原地加(改变会带到外面)，不能使用p = p - lr * GradP(只改变局部变量)

        return

# 定义带有动量的SGD
'''
加速收敛：
    在梯度方向一致的维度上累积速度
    类似球下坡时越滚越快
减少震荡：
    动量作为低通滤波器
    平滑梯度更新方向
    在梯度方向变化大的维度上抑制震荡
逃离局部最优：
    动量积累的"惯性"可越过狭窄的局部极小点
    特别有利于逃离鞍点区域
'''
class MySGDWithMomentum(Optimizer):
    # 初始化函数，设置优化器的超参数和需要更新的参数
    def __init__(self, params, lr=0.01, momentum=0 , dampening=0):
        #momentum是动量，dampening是阻尼

        # 检查学习率是否合法（不能为负数）
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        # 将所有超参数放到一个字典中，便于后续使用
        defaults = dict(lr=lr , momentum=momentum , dampening=dampening)
        
        # 调用父类 Optimizer 的构造函数，传入参数组和默认超参数
        super().__init__(params, defaults)#父类Optimizer会将params, defaults传入self的param_groups中

    # 使用 @torch.no_grad() 装饰器避免在更新时计算梯度
    @torch.no_grad()
    def step(self):
        momentum = group['momentum']
        dampening = group['dampening']

        # 遍历所有参数组（一个优化器可以管理多个参数组）
        for group in self.param_groups:
            # 遍历当前参数组中的所有参数
            for p in group['params']:
                if p.grad is None:# 如果参数没有梯度，则跳过更新
                    continue

                GradP = p.grad # 获取参数的梯度(∂loss / ∂p)
                # 注意这里p.grad是一个tensor值，此时GradP与p.grad指向同一个内存地址

                # 从优化器状态中获取当前参数对应的状态字典
                param_state = self.state[p]
                # 如果动量不为 0，则进行动量相关的更新
                if momentum != 0:
                    # momentum_buffe中存的是历史梯度信息

                    
                    if 'momentum_buffer' not in param_state:
                        # 如果状态中没有存储动量缓冲区，则初始化一个动量缓冲区（克隆当前梯度）

                        buf = torch.clone(GradP)#创建GradP张量的一个独立副本
                        #注意buf不同于GradP，他指向一个新的内存地址，此时的值等于GradP指向的内存地址的值

                        buf.detach()#将克隆的张量从计算图中分离,使动量缓冲区不参与梯度计算
                        param_state['momentum_buffer'] = buf
                    else:
                        # 如果已经存在动量缓冲区，则取出该缓冲区

                        buf = param_state['momentum_buffer']

                        # 按照动量公式更新：buf = momentum * buf + (1 - dampening) * GradP
                        buf.mul_(momentum).add_(GradP, alpha=1 - dampening)

                        GradP = buf

                # 根据更新公式，使用梯度下降：p = p - lr * GradP
                p.add_(GradP, alpha=-group['lr'])#注意这里add_是原地加(改变会带到外面)，不能使用p = p - lr * GradP(只改变局部变量)

        return

