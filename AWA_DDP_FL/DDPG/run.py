import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from DDPG.parsers import args
from DDPG.RL_brain import ReplayBuffer, DDPG


seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# -------------------------------------- #
# 环境加载
# -------------------------------------- #
class CustomEnv:
    def __init__(self):
        self.state = None
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.Time = 0
        self.max_Time = 50
        self.penalty_coefficient = 20000
    def reset(self, remind_budget):
        self.state = np.array([6/20])  # 假设的初始状态
        self.remind_budget = remind_budget
        self.Time = 0
        return self.state

    def step(self, action, D, Epoch, global_epoch, acc):
        # 这里添加你的环境逻辑
        self.max_Time = Epoch
        next_state = np.zeros(1)
        #print(action)
        done = False
        excess_penalty = 0
        next_state[0] = (self.state[0]*20 + (0.01+(151-Epoch)/40)*action)/20
        self.state = next_state
        reward1 = -100 * (1-(0.01+global_epoch/300))*(np.square(self.state[0] * 20) - ((7.5 * 0.36) / (min(0.36, 10*acc)*max(0.9,D))) * (self.state[0] * 20) + 0)
        if global_epoch < 30:
            reward2 = -0.0005*(155-Epoch)* np.square((self.state[0]*20) * (Epoch - self.Time) - self.remind_budget)
        elif 30 <= global_epoch < 80:
            reward2 = -0.002 * (1+(global_epoch-60)*99/120) * np.square((self.state[0] * 20) * (Epoch - self.Time) - self.remind_budget)
        elif 80 <= global_epoch < 120:
            reward2 = -1 * (155 - Epoch) * np.square((self.state[0] * 20) * (Epoch - self.Time) - self.remind_budget)
        elif 120 <= global_epoch < 130:
            reward2 = -0.01 * (155 - Epoch) * np.square((self.state[0] * 20) * (Epoch - self.Time) - self.remind_budget)
        else:
            reward2 = -400 * np.square((self.state[0] * 20) * (Epoch - self.Time) - self.remind_budget)

        self.remind_budget = self.remind_budget - self.state[0]*20

        if next_state[0] <0 :
            excess_penalty = excess_penalty-10000
        if self.remind_budget <0 :
            excess_penalty = excess_penalty-10000
            #print(self.Time,"状态2超阈值", next_state[1])
        reward = reward1 + reward2 + excess_penalty
        if self.Time == self.max_Time:
            done = True
            #print(self.state[0])
            #print(self.state[1])
        self.Time = self.Time + 1
        #self.state = next_state
        return next_state, reward, done

    def render(self, mode='human'):
        print(f"State: {self.state}")

    def close(self):
        pass

env = CustomEnv()
n_states = env.observation_space.shape[0]  # 状态数 2
n_actions = env.action_space.shape[0]  # 动作数 1
action_bound = env.action_space.high[0]  # 动作的最大值 1.0
# -------------------------------------- #
# 模型构建
# -------------------------------------- #
def train(D, Epoch, global_epoch, remind_budget, acc, combination):
# 经验回放池实例化
    action_best_next_state=[]
    agents = []
    replay_buffers = []
    for _ in range(10):
        agent = DDPG(n_states=n_states,  # 状态数
                     n_hiddens=args.n_hiddens,  # 隐含层数
                     n_actions=n_actions,  # 动作数
                     action_bound=action_bound,  # 动作最大值
                     sigma=args.sigma,  # 高斯噪声
                     actor_lr=args.actor_lr,  # 策略网络学习率
                     critic_lr=args.critic_lr,  # 价值网络学习率
                     tau=args.tau,  # 软更新系数
                     gamma=args.gamma,  # 折扣因子
                     device=device
                     )
        replay_buffer = ReplayBuffer(capacity=args.buffer_size)
        agents.append(agent)
        replay_buffers.append(replay_buffer)

    # -------------------------------------- #
    # 模型训练
    # -------------------------------------- #

    return_list = []  # 记录每个回合的return
    mean_return_list = []  # 记录每个回合的return均值
    for k in range(10):
        if k not in combination:
            action_best_next_state.append(6)
            print("跳过")
            continue
        agent = agents[k]
        replay_buffer = replay_buffers[k]
        return_list = []  # 记录每个回合的return
        mean_return_list = []  # 记录每个回合的return均值
        best_next_state = None  # 存储奖励值最大的下一时刻的状态值
        max_reward = -float('inf')  # 初始化最大奖励值
        max_r = -float('inf')
        State = np.zeros((150,100))
        R = np.zeros((150,100))
        for i in range(150):  # 迭代10回合
            episode_return = 0  # 累计每条链上的reward
            state = env.reset(remind_budget[k])  # 初始时的状态
            done = False  # 回合结束标记
            t=0
            while not done:
                # 获取当前状态对应的动作
                action = agent.take_action(state)
                # 环境更新
                next_state, reward, done = env.step(action, D[k], 100 - Epoch, global_epoch, acc[k])
                # 更新经验回放池
                replay_buffer.add(state, action, reward, next_state, done)
                # 状态更新
                state = next_state
                State[i][t] = state[0]
                # 累计每一步的reward
                episode_return += reward
                R[i][t] = reward
                # 如果经验池超过容量，开始训练
                if replay_buffer.size() > args.min_size:
                    # 经验池随机采样batch_size组
                    s, a, r, ns, d = replay_buffer.sample(args.batch_size)
                    # 构造数据集
                    transition_dict = {
                        'states': s,
                        'actions': a,
                        'rewards': r,
                        'next_states': ns,
                        'dones': d,
                    }
                    # 模型训练
                    agent.update(transition_dict)
                t = t + 1
            if episode_return > max_reward:
                max_reward = episode_return
                best_it = i
            # 保存每一个回合的回报
            return_list.append(episode_return)
            mean_return_list.append(np.mean(return_list[-10:]))  # 平滑
        for ii in range(0,100-Epoch):
            if R[best_it][ii] > max_r:
                max_r = R[best_it][ii]
                best_next_state = 20*State[best_it][ii]
        action_best_next_state.append(best_next_state)
        print("number:", k, best_next_state)
    return action_best_next_state

