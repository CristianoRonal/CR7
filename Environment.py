import matlab.engine
import numpy as np
import matplotlib.pyplot as plt
import random
import gym
from gym import spaces


class CruiseControlEnvironment(gym.Env):

    def __init__(self):
        super(CruiseControlEnvironment, self).__init__()
        self.eng = matlab.engine.start_matlab()  # 启动matlab环境
        # 动作空间维度
        # self.action_space_n = 1
        # self.observation_space_n = 6   # 本车速度、本车加速度、本车位置、与前车速度差、与前车距离、期望预稳速度
        # 动作的范围
        # self.action_space_low = -3.0
        # self.action_space_high = 2.0
        # self.action_space = np.array([self.action_space_low, self.action_space_high])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)  # 本车速度、本车加速度、本车位置、与前车速度差、与前车距离、期望预稳速度
        self.action_space = spaces.Box(low=-3.0, high=2.0, shape=(1,), dtype=np.float32)
        # self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)   # weight权重
        # 总时间Tf，采样时间Ts
        self.Ts = 0.1
        self.Tf = 200
        self.max_steps = int(self.Tf/self.Ts)


    def seed(self, seed=None):
        """设置环境的随机种子，同时同步到 MATLAB"""
        if seed is not None:
            self.eng.rng(float(seed), nargout=0)  # 设置固定种子
            random.seed(seed)
            np.random.seed(seed)
            return [seed]
        # 如果不提供种子，不设置固定值，保持随机性
        return []

    ''' 初始化状态重置'''
    def reset(self, seed=None, options=None):
        """重置环境，每次都使用纯粹随机种子"""
        if seed is not None:
            self.seed(seed)  # 如果提供种子，则使用固定种子
        else:
            self.eng.rng('shuffle', nargout=0)  # 每次 reset 时基于时间随机化
        # self.eng.eval()是MATLAB Engine API for Python提供的一个方法，可以理解为在matlab的命令行窗口执行command
        # nargout=0说明不期望该命令有任何返回值，类似于在 MATLAB 命令行窗口中运行一个不返回任何输出的命令（例如 clear 或 clc）
        self.eng.eval('Reset', nargout=0)   # 执行了Reset.m
        # 取所有状态
        state = self.eng.eval('observation')
        # 取最后一个状态，格式转换
        state = np.array(state[-1]).astype(np.float32)
        state = state.reshape(6)
        # print(state)
        return state

    ''' 步进执行 '''
    def step(self, action):
        # 连续系统，步进执行，返回s_、r、isdone等参数情况
        # 通过在中途改变模块的值来间接改变输入值
        # reward定义在step()函数下
        # a_acc = self.eng.eval('a_acc')
        # if isinstance(a_acc, float) == True:
        #     a_acc = np.array(a_acc).astype(np.float32)
        #     a_acc = a_acc.reshape(1)
        # else:
        #     a_acc = np.array(a_acc[-1]).astype(np.float32)
        #     a_acc = a_acc.reshape(1)
        # a_plan = self.eng.eval('a_plan')
        # if isinstance(a_plan, float) == True:
        #     a_plan = np.array(a_plan).astype(np.float32)
        #     a_plan = a_plan.reshape(1)
        # else:
        #     a_plan = np.array(a_plan[-1]).astype(np.float32)
        #     a_plan = a_plan.reshape(1)
        # weight = float(action)
        # a_exe = weight*a_plan + (1-weight)*a_acc
        
        # self.eng.workspace['acceleration'] = float(a_exe)  # 将action 写入workspace中
        self.eng.workspace['acceleration'] = float(action)  # 将action 写入workspace中
        # self.eng.eval("set_param('DRL_ACC/Gain', 'Gain', num2str(acceleration))", nargout=0)
        self.eng.eval("set_param('e2eDRL_ACC/Gain', 'Gain', num2str(acceleration))", nargout=0)
        # self.eng.eval("set_param('Expert_data/Gain', 'Gain', num2str(acceleration))", nargout=0)
        self.eng.eval("set_param(model_name, 'SimulationCommand', 'step')", nargout=0)
        next_state = self.eng.eval('observation')
        reward = self.eng.eval('reward')
        done = self.eng.eval('isdone')
        next_state = np.array(next_state[-1]).astype(np.float32)
        next_state = next_state.reshape(6)
        reward = np.array(reward[-1]).astype(np.float32) 
        reward = reward.reshape(1)   
        done = np.array(done[-1]).astype(bool) 
        done = done.reshape(1)        
        return next_state, reward, done, {}        # 参照gym要求返回四个值：next_state, reward, done, info。返回info为空字典{}

    ''' 获取a_acc和a_plan '''
    def gainprior(self):
        a_acc = self.eng.eval('a_acc')
        if isinstance(a_acc, float) == True:
            a_acc = np.array(a_acc).astype(np.float32)
            a_acc = a_acc.reshape(1)
        else:
            a_acc = np.array(a_acc[-1]).astype(np.float32)
            a_acc = a_acc.reshape(1)
        a_plan = self.eng.eval('a_plan')
        if isinstance(a_plan, float) == True:
            a_plan = np.array(a_plan).astype(np.float32)
            a_plan = a_plan.reshape(1)
        else:
            a_plan = np.array(a_plan[-1]).astype(np.float32)
            a_plan = a_plan.reshape(1)
        return a_acc, a_plan
    
    def gain_vdes(self):
        v_des = self.eng.eval('v_des')
        return v_des
    
    def gain_Reward(self):
        R_headway = self.eng.eval('R_headway')
        if isinstance(R_headway, float) == True:
            R_headway = np.array(R_headway).astype(np.float32)
            R_headway = R_headway.reshape(1)
        else:
            R_headway = np.array(R_headway[-1]).astype(np.float32)
            R_headway = R_headway.reshape(1)
        R_comfort = self.eng.eval('R_comfort')
        if isinstance(R_comfort, float) == True:
            R_comfort = np.array(R_comfort).astype(np.float32)
            R_comfort = R_comfort.reshape(1)
        else:
            R_comfort = np.array(R_comfort[-1]).astype(np.float32)
            R_comfort = R_comfort.reshape(1)
        return R_headway, R_comfort

    ''' 停止仿真 '''
    def stop(self):
        self.eng.eval("set_param(model_name, 'SimulationCommand', 'stop')", nargout=0)
        # 清屏，清除工作空间的变量，方便下一次重新启动
        self.eng.eval("clc", nargout=0)
        self.eng.eval("clear", nargout=0)

    ''' 随机选择动作输出 '''
    # def action_sample(self):
    #     action = np.random.uniform(self.action_space_low, self.action_space_high, self.action_space_n)
    #     return action

    ''' 退出Matlab-simulink释放进程 '''
    def exit(self):
        self.eng.quit()
        self.eng.exit()
