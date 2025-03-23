import pickle
import numpy as np
from Environment import CruiseControlEnvironment  # 替换为你的环境
import matplotlib.pyplot as plt
import torch
from models.mlp_policy import Policy
from core.agent import Agent

# 加载文件
# file_path = './datasets/converted_expert_traj.p'
# with open(file_path, 'rb') as f:
#     expert_traj, running_state = pickle.load(f)

# 获取维度
env = CruiseControlEnvironment()
state_dim = env.observation_space.shape[0]  # 状态维度
action_dim = env.action_space.shape[0]      # 动作维度

# # 提取动作
# actions = expert_traj[:, state_dim:]
# print("expert_traj 形状:", expert_traj.shape)
# print("actions 形状:", actions.shape)

# # 检查动作范围
# low = env.action_space.low
# high = env.action_space.high
# print("动作范围:", actions.min(), "到", actions.max())
# print("环境动作范围:", low, "到", high)

# 可视化动作
# plt.plot(actions)
# plt.title("Actions over Time")
# plt.xlabel("Timestep")
# plt.ylabel("Action Value")
# plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = Policy(state_dim, action_dim).to(device)
agent = Agent(policy=policy, device=device, num_threads=1)

# 收集样本并检查
batch, log = agent.collect_samples(min_batch_size=1000)
print("Action Statistics:")
print("  Mean:", log['action_mean'])
print("  Min:", log['action_min'])
print("  Max:", log['action_max'])