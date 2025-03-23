import numpy as np
import torch
from Environment import CruiseControlEnvironment
import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *

# 加载 mpc_expert_dataset
dataset = torch.load('./datasets/mpc_expert_dataset2.pt', weights_only=False)

# 检查数据集结构
assert 'states' in dataset and 'actions' in dataset, "数据集缺少 'states' 或 'actions'"
total_timesteps, state_dim = dataset['states'].shape
action_dim = dataset['actions'].shape[1]
assert dataset['actions'].shape[0] == total_timesteps, "状态和动作的时间步长不一致"

# 初始化 running_state
running_state = ZFilter((state_dim,), clip=5)

# 更新 running_state 的统计量
for state in dataset['states']:
    running_state(state, update=True)
running_state.fix = True  # 固定统计量

# 标准化状态
standardized_states = np.array([running_state(state, update=False) for state in dataset['states']])

# 反归一化动作
env = CruiseControlEnvironment()
low = env.action_space.low
high = env.action_space.high
assert action_dim == len(low) == len(high), "动作维度与环境不匹配"
actions_raw = (dataset['actions'] + 1) / 2 * (high - low) + low

# 拼接为 expert_traj
expert_traj = np.hstack([standardized_states, actions_raw])

# 保存
import pickle
with open('./datasets/converted_expert_traj.p', 'wb') as f:
    pickle.dump((expert_traj, running_state), f)

print("转换完成, expert_traj 形状：", expert_traj.shape)