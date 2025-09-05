import torch
import torch.optim as optim
import copy
import math
from quantization import LatticeQuantization, ScalarQuantization
from privacy import Privacy
from data_generation import PV_data
from torch.utils.data import Dataset
import numpy as np
import time
class PVLoadDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

def federated_setup(global_model, train_data, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    indexes = torch.randperm(len(train_data))
    user_data_len = math.floor(len(train_data) / args.num_users)
    local_models = {}
    for user_idx in range(args.num_users):
        user = {'data': torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_data,
                                    indexes[user_idx * user_data_len:(user_idx + 1) * (user_data_len-550)]),
            batch_size=args.train_batch_size, shuffle=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models


def federated_setup2(global_model, args):
    # create a dict of dict s (local users), i.e. {'1': {'data':..., 'model':..., 'opt':...}, ...}
    #indexes = torch.randperm(len(train_data))
    #user_data_len = math.floor(len(train_data) / args.num_users)
    local_models = {}
    for user_idx in range(args.num_users):
        train_x, train_y = PV_data(client=user_idx, dataset_type='train')
        train_ds = PVLoadDataset(train_x, train_y)

        user = {'data': torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.train_batch_size, shuffle=False, drop_last=True),
            'model': copy.deepcopy(global_model)}
        user['opt'] = optim.SGD(user['model'].parameters(), lr=args.lr,
                                momentum=args.momentum) if args.optimizer == 'sgd' \
            else optim.Adam(user['model'].parameters(), lr=args.lr)
        if args.lr_scheduler:
            user['scheduler'] = optim.lr_scheduler.ReduceLROnPlateau(user['opt'], patience=10, factor=0.1, verbose=True)
        local_models[user_idx] = user
    return local_models

def distribute_model(local_models, global_model):
    for user_idx in range(len(local_models)):
        local_models[user_idx]['model'].load_state_dict(copy.deepcopy(global_model.state_dict()))






def aggregate3(local_models, global_model, combination, weight, eplison, args):  # FedAvg
    """
    普通联邦聚合（FedAvg）
    参数:
        local_models: 本地模型列表
        global_model: 全局模型
        combination: 参与聚合的用户索引列表
    """
    # 创建全局模型状态字典的深拷贝
    state_dict = copy.deepcopy(global_model.state_dict())

    # 遍历模型的所有层
    for key in state_dict.keys():
        # 初始化平均权重张量
        local_weights_average = torch.zeros_like(state_dict[key])

        # 对每个参与聚合的用户进行平均
        for idx_comb, user_idx in enumerate(combination):
            # 获取本地模型权重
            local_weight = local_models[user_idx]['model'].state_dict()[key]

            # 累加本地权重
            local_weights_average += weight[idx_comb]*add_noise(local_weight, eplison[user_idx], args)

        # 计算平均权重（简单平均）
        state_dict[key] = local_weights_average
    # 加载更新后的状态字典到全局模型
    global_model.load_state_dict(state_dict)

    return None  # 普通FedAvg不需要返回SNR








class JoPEQ:  # Privacy Quantization class
    def __init__(self, args):
        self.vec_normalization = args.vec_normalization
        dither_var = None
        if args.quantization:
            if args.lattice_dim > 1:
                self.quantizer = LatticeQuantization(args)
                dither_var = self.quantizer.P0_cov
            else:
                self.quantizer = ScalarQuantization(args)
                dither_var = (self.quantizer.delta ** 2) / 12
        else:
            self.quantizer = None
        if args.privacy:
            self.privacy = Privacy(args, dither_var)
        else:
            self.privacy = None

    def divide_into_blocks(self, input, dim=2):
        # Zero pad if needed
        modulo = len(input) % dim
        if modulo:
            pad_with = dim - modulo
            input_vec = torch.cat((input, torch.zeros(pad_with).to(input.dtype).to(input.device)))
        else:
            pad_with = 0
        input_vec = input.view(dim, -1)  # divide input into blocks
        return input_vec, pad_with,

    def __call__(self, input):
        original_shape = input.shape
        input = input.view(-1)
        if self.vec_normalization:  # normalize
            input, pad_with = self.divide_into_blocks(input)

        mean = torch.mean(input, dim=-1, keepdim=True)
        std = torch.norm(input - mean) / (input.shape[-1] ** 0.5)

        std = 3 * std
        input = (input - mean) / std

        if self.privacy is not None:
            input = self.privacy(input)

        if self.quantizer is not None:
            input = self.quantizer(input)

        # denormalize
        input = (input * std) + mean

        if self.vec_normalization:
            input = input.view(-1)[:-pad_with] if pad_with else input  # remove zero padding

        input = input.reshape(original_shape)

        return input

def compute_euclidean_distance(model1, model2):
    # 获取两个模型的参数
    params1 = model1
    params2 = model2

    # 确保两个模型的参数数量一致
    assert len(params1) == len(params2), "Models should have the same number of parameters"
    # 初始化欧氏距离
    distance = 0.0
    # 逐层计算每一层参数的欧氏距离
    for p1, p2 in zip(params1, params2):
        # 将每个参数展平成一维
        p1 = p1.view(-1)
        p2 = p2.view(-1)

        # 计算每一层参数的欧氏距离并累加
        distance += torch.sum((p1 - p2) ** 2)
    # 返回最终的欧氏距离
    return torch.sqrt(distance).to("cuda:0")


def add_noise(local_weights, eplison, args):
    sensitivity = cal_sensitivity(args.lr, 10, args.train_batch_size)
    noise_scale = np.sqrt(2 * np.log(1.25 / 0.001)) / eplison
    local_weights += torch.from_numpy(np.random.normal(loc=0, scale=sensitivity * noise_scale,
                                                        size=local_weights.shape)).to(args.device)
    return local_weights

def cal_sensitivity(lr, clip, dataset_size):
    return 2 * lr * clip / dataset_size