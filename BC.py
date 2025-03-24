import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pickle
import os
from torch.utils.tensorboard import SummaryWriter
from models.mlp_policy import Policy  # 重用您GAIL代码中的Policy类
from utils import *  # 假设包括assets_dir()等工具函数

# 参数解析器（与GAIL代码保持一致）
import argparse

parser = argparse.ArgumentParser(description='PyTorch行为克隆用于策略预训练')
parser.add_argument('--env-name', default="Previewed Cruise Control", metavar='G',
                    help='运行的环境名称')
parser.add_argument('--expert-traj-path', metavar='G', 
                    default='./datasets/converted_expert_traj.p',
                    help='专家轨迹的路径')
parser.add_argument('--log-std', type=float, default=-0.0, metavar='G',
                    help='策略的对数标准差（默认: -0.0）')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='学习率（默认: 3e-4）')
parser.add_argument('--gpu-index', type=int, default=0, metavar='N',
                    help='GPU索引（默认: 0）')
args = parser.parse_args()

# 设备和数据类型设置
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

# TensorBoard记录器
writer = SummaryWriter(log_dir='./runs/bc_pretraining')

def train_bc():
    """
    使用行为克隆方法训练Policy模型, 考虑专家数据的标准化。
    """
    # 加载专家数据和标准化器
    expert_traj, running_state = pickle.load(open(args.expert_traj_path, "rb"))
    running_state.fix = True  # 确保标准化参数不再更新，与GAIL一致
    
    # 提取状态和动作维度
    state_dim = expert_traj.shape[1] - 1  # 假设最后一列是动作
    action_dim = 1  # 根据您的GAIL代码，动作维度为1
    
    # 将expert_traj拆分为状态和动作
    states = expert_traj[:, :state_dim]  # 已标准化的状态
    actions = expert_traj[:, state_dim:]
    
    # 转换为PyTorch张量
    states = torch.from_numpy(states).to(dtype).to(device)
    actions = torch.from_numpy(actions).to(dtype).to(device)
    
    # 定义Policy网络（与GAIL中一致）
    policy_net = Policy(state_dim, action_dim, log_std=args.log_std).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
    
    # 创建DataLoader用于批量训练
    dataset = TensorDataset(states, actions)
    batch_size = 64  # 可调整的批量大小
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 训练参数
    num_epochs = 100  # 训练轮数，可调整
    print_interval = 10  # 每10轮打印一次信息
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (states_batch, actions_batch) in enumerate(loader):
            # 前向传播：获取预测的动作均值
            action_mean, _, _ = policy_net(states_batch)
            loss = criterion(action_mean, actions_batch)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # 计算每轮的平均损失
        avg_loss = total_loss / num_batches
        
        # 记录到TensorBoard
        writer.add_scalar('BC/Avg_Loss', avg_loss, epoch)
        
        # 打印训练信息
        if (epoch + 1) % print_interval == 0:
            print(f'轮次 [{epoch + 1}/{num_epochs}]，平均损失: {avg_loss:.6f}')
    
    # 保存训练好的模型和标准化器
    model_save_path = os.path.join(assets_dir(), 'learned_models', f'{args.env_name}_bc.p')
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # 确保目录存在
    pickle.dump((policy_net, running_state), open(model_save_path, 'wb'))  # 保存policy_net和running_state
    print(f'预训练的BC Policy模型和标准化器已保存至 {model_save_path}')
    
    # 关闭TensorBoard记录器
    writer.close()

if __name__ == '__main__':
    train_bc()