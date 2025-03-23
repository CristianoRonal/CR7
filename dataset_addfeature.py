import torch
import numpy as np
import pandas as pd

# 参数设置
dataset_file = './datasets/mpc_expert_dataset.pt'  # 原数据集路径
states_df = pd.read_csv('states_df.csv')
new_feature = np.array(states_df['v_des'].to_list())  # 替换为你的新维度数据（长度需与 states 的行数一致）

# 1. 加载原始数据集
dataset = torch.load(dataset_file, weights_only=False)
states = dataset['states']
print("原始 states 形状：", states.shape)

# 2. 检查新数据长度是否匹配
if len(new_feature) != states.shape[0]:
    raise ValueError(f"新数据的长度 ({len(new_feature)}) 与 states 的行数 ({states.shape[0]}) 不匹配")

# 3. 将新数据转为 2D 列向量（如果需要）
if new_feature.ndim == 1:
    new_feature = new_feature.reshape(-1, 1)

# 4. 增加新维度并合并
new_states = np.column_stack([states, new_feature])
dataset['states'] = new_states
print("新 states 形状：", dataset['states'].shape)

# 5. 保存到新位置
torch.save(dataset, './datasets/mpc_expert_dataset2.pt')
print(f"更新后的 dataset 已保存")