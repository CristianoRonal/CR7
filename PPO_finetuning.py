import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import sys
import time
from torch.utils.tensorboard import SummaryWriter
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Environment import CruiseControlEnvironment  # 假设环境文件名为Environment.py
from utils import to_device, ZFilter, LongTensor  # 使用您的utils模块
from models.mlp_policy import Policy
from models.mlp_critic import Value
from core.ppo import ppo_step
from core.common import estimate_advantages
from core.agent import Agent

# 参数解析器
parser = argparse.ArgumentParser(description='PyTorch PPO Fine-tuning with Pretrained BC Policy')
parser.add_argument('--env-name', default="Previewed Cruise Control", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--pretrained-model-path', metavar='G', 
                    default='./assets/learned_models/Previewed Cruise Control_bc.p',
                    help='path of the pretrained BC model')
parser.add_argument('--render', action='store_true', default=False,
                    help='render the environment')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae parameter (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=3e-4, metavar='G',
                    help='learning rate (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='G',
                    help='clipping epsilon for PPO (default: 0.2)')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size per PPO update (default: 2048)')
parser.add_argument('--eval-batch-size', type=int, default=2048, metavar='N',
                    help='minimal batch size for evaluation (default: 2048)')
parser.add_argument('--max-iter-num', type=int, default=500, metavar='N',
                    help='maximal number of main iterations (default: 500)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 1)')
parser.add_argument('--save-model-interval', type=int, default=20, metavar='N',
                    help="interval between saving model (default: 20)")
parser.add_argument('--gpu-index', type=int, default=0, metavar='N',
                    help='GPU index (default: 0)')
args = parser.parse_args()

# 设备和数据类型设置
dtype = torch.float32
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

# TensorBoard记录器
writer = SummaryWriter(log_dir='./runs/ppo_finetuning3')

# 加载预训练的策略网络和标准化器
policy_net, running_state = pickle.load(open(args.pretrained_model_path, "rb"))
policy_net.to(device)

# 定义值网络
state_dim = policy_net.affine_layers[0].in_features  # 从策略网络获取状态维度
action_dim = policy_net.action_mean.out_features  # 从策略网络获取动作维度
value_net = Value(state_dim, hidden_size=(128, 128), activation='tanh').to(device)

# 定义优化器
optimizer_policy = optim.Adam(policy_net.parameters(), lr=args.learning_rate)
optimizer_value = optim.Adam(value_net.parameters(), lr=args.learning_rate)

# PPO优化参数
optim_epochs = 10  # PPO更新时的优化轮数
optim_batch_size = 64  # 每个优化批次的大小

# 创建环境
env = CruiseControlEnvironment()

# 创建Agent
agent = Agent(env, policy_net, device, running_state=running_state, num_threads=args.num_threads)

def update_params(batch, i_iter):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    
    # 计算值网络预测
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    # 获取优势估计和回报
    advantages, returns = estimate_advantages(rewards, masks, values, args.gamma, args.tau, device)

    # 进行mini-batch PPO更新（参考GAIL的更新方式）
    optim_iter_num = int(np.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = \
            states[perm].clone(), actions[perm].clone(), returns[perm].clone(), advantages[perm].clone(), fixed_log_probs[perm].clone()

        for i in range(optim_iter_num):
            ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0]))
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = \
                states[ind], actions[ind], advantages[ind], returns[ind], fixed_log_probs[ind]

            ppo_step(policy_net, value_net, optimizer_policy, optimizer_value, 1, states_b, actions_b, returns_b,
                     advantages_b, fixed_log_probs_b, args.clip_epsilon, args.l2_reg)

def main_loop():
    for i_iter in range(args.max_iter_num):
        # 收集样本
        to_device(torch.device('cpu'), policy_net)  # 采样时移到CPU
        batch, log = agent.collect_samples(args.min_batch_size, render=args.render)
        to_device(device, policy_net)  # 更新时移回GPU

        t0 = time.time()
        update_params(batch, i_iter)
        t1 = time.time()

        # 评估（使用确定性动作）
        # to_device(torch.device('cpu'), policy_net)
        # _, log_eval = agent.collect_samples(args.eval_batch_size, mean_action=True)
        # to_device(device, policy_net)
        t2 = time.time()

        # 记录到TensorBoard和打印日志
        writer.add_scalar('PPO/Avg_Reward', log['avg_reward'], i_iter)
        writer.add_scalar('PPO/Max_Reward', log['max_reward'], i_iter)
        writer.add_scalar('PPO/Min_Reward', log['min_reward'], i_iter)
        # writer.add_scalar('PPO/Eval_Avg_Reward', log_eval['avg_reward'], i_iter)

        if i_iter % args.log_interval == 0:
            print('{}\tT_sample {:.4f}\tT_update {:.4f}\ttrain_R_avg {:.2f}'.format(
                i_iter, log['sample_time'], t1-t0, float(log['avg_reward'])))

        # 保存模型
        # if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
        #     to_device(torch.device('cpu'), policy_net, value_net)
        #     save_path = os.path.join('assets/learned_models', f'{args.env_name}_ppo_iter{i_iter + 1}_2.p')
        #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
        #     pickle.dump((policy_net, value_net, running_state), open(save_path, 'wb'))
        #     to_device(device, policy_net, value_net)
        # 保存模型（torch.save)
        if args.save_model_interval > 0 and (i_iter + 1) % args.save_model_interval == 0:
            to_device(torch.device('cpu'), policy_net, value_net)
            save_path = os.path.join('assets/learned_models', f'{args.env_name}_ppo_iter{i_iter + 1}_3.pth')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'policy_net': policy_net.state_dict(),
                'value_net': value_net.state_dict(),
                'running_state': running_state
            }, save_path)
            to_device(device, policy_net, value_net)

        # 清理GPU内存
        torch.cuda.empty_cache()

    # 保存最终模型
    # to_device(torch.device('cpu'), policy_net, value_net)
    # final_save_path = os.path.join('assets/learned_models', f'{args.env_name}_ppo_final_2.p')
    # pickle.dump((policy_net, value_net, running_state), open(final_save_path, 'wb'))
    # print(f'最终PPO模型已保存至 {final_save_path}')

    # 保存最终模型（torch.save）
    to_device(torch.device('cpu'), policy_net, value_net)
    final_save_path = os.path.join('assets/learned_models', f'{args.env_name}_ppo_final_3.pth')
    torch.save({
        'policy_net': policy_net.state_dict(),
        'value_net': value_net.state_dict(),
        'running_state': running_state
    }, final_save_path)
    print(f'最终PPO模型已保存至 {final_save_path}')

    # 清理环境
    # env.stop()
    # env.exit()

# 运行主循环
main_loop()

# 关闭TensorBoard记录器
writer.close()