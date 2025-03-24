import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["OMP_NUM_THREADS"] = "1"  # 设置线程数为1，避免多线程问题
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决可能的库冲突

import torch
torch.set_num_threads(1)  # PyTorch线程数设置为1
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端生成图像
import matplotlib.pyplot as plt
from Environment import CruiseControlEnvironment  # 导入巡航控制环境
from models.mlp_policy import Policy  # 导入策略网络模型
from utils import ZFilter  # 导入状态标准化工具

# 设置设备（优先使用GPU）
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_path = 'assets/learned_models/Previewed Cruise Control_bc.p'  # BC模型路径
MAX_STEPS = 2000  # 每回合最大步数
NUM_EPISODES = 10  # 测试回合数

def load_model_and_running_state(model_path):
    """加载预训练的BC策略模型和运行状态"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到：{model_path}")
    with open(model_path, 'rb') as f:
        policy_net, running_state = pickle.load(f)  # 加载策略网络和标准化器
    policy_net.to(device)  # 将模型移到指定设备
    policy_net.eval()  # 设置为评估模式
    print("策略模型和运行状态加载成功！")
    return policy_net, running_state

def plot_test(ego_v, ego_dp, ego_x, i_episode):
    """绘制速度和距离差与位置的关系图"""
    ego_v = np.array(ego_v) * 3.6  # 速度转换为 km/h
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(ego_x, ego_v, label='速度')
    ax1.set_xlim(0, 1100)  # X轴范围
    ax1.set_ylim(0, 130)  # Y轴范围
    ax1.set_title('速度 vs. 位置')
    ax1.set_xlabel('位置 (m)')
    ax1.set_ylabel('速度 (km/h)')
    ax1.legend()
    ax2.plot(ego_x, ego_dp, label='距离差')
    ax2.set_xlim(0, 1100)
    ax2.set_title('距离差 vs. 位置')
    ax2.set_xlabel('位置 (m)')
    ax2.set_ylabel('距离差 (m)')
    ax2.legend()
    plt.tight_layout()
    os.makedirs('./figure', exist_ok=True)  # 确保保存目录存在
    plt.savefig(f'./figure/bc_test_eps{i_episode}.png')  # 保存图像
    plt.close(fig)

def test_model():
    """在环境中测试BC策略"""
    policy_net, running_state = load_model_and_running_state(model_path)
    env = CruiseControlEnvironment()  # 初始化巡航控制环境

    for eps in range(NUM_EPISODES):
        raw_state = env.reset()  # 获取原始初始状态
        state = running_state(raw_state)  # 标准化初始状态
        episode_reward = 0  # 回合总奖励
        ego_v, ego_dp, ego_x = [], [], []  # 存储状态数据用于绘图
        v_des = env.gain_vdes() if hasattr(env, 'gain_vdes') else 0  # 获取期望速度（如果有）

        print(f"回合 {eps} - 初始原始状态: {raw_state}")
        print(f"回合 {eps} - 初始标准化状态: {state}")

        for step in range(MAX_STEPS):
            # 将状态转换为张量并生成动作
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_mean, _, _ = policy_net(state_tensor)  # 获取动作均值
                action = action_mean.cpu().numpy()[0]  # 使用均值作为动作

            # 在环境中执行一步
            raw_next_state, reward, done, _ = env.step(action)
            next_state = running_state(raw_next_state)  # 标准化下一状态

            # 记录原始状态数据用于绘图
            ego_v.append(raw_state[0])  # 速度
            ego_dp.append(raw_state[4])  # 距离差
            ego_x.append(raw_state[2])  # 位置

            # 每100步打印调试信息
            if step % 100 == 0:
                print(f"步数 {step} - 动作: {action}, 原始状态: {raw_state}, 奖励: {reward}")

            episode_reward += reward
            state = next_state
            raw_state = raw_next_state

            if done or step == (MAX_STEPS - 1):
                if hasattr(env, 'stop'):
                    env.stop()  # 停止环境（如果支持）
                break

        print(f"回合 {eps} - 最终速度数据长度: {len(ego_v)}, 位置范围: {min(ego_x)} 到 {max(ego_x)}")
        plot_test(ego_v, ego_dp, ego_x, eps)  # 绘制结果
        print(f'回合: {eps} | 回合奖励: {episode_reward} | 期望速度: {v_des*3.6} km/h')

    env.close()  # 关闭环境

if __name__ == "__main__":
    test_model()