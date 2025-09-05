import gc
import sys
import time
import copy
import random
from statistics import mean

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from torchinfo import summary
from opacus.grad_sample import GradSampleModule

import utils
import models
import federated_utils
from configurations import args_parser
from attention import FederatedLearningWithAttention
from DDPG.run import train
import warnings
warnings.filterwarnings("ignore")

# ==========================
# 工具函数
# ==========================
def normalize(values, indices):
    """归一化并加上偏移量，避免除 0"""
    raw = [values[i] - min(values[x] for x in indices) for i in indices]
    max_val = max(raw) or 1
    return [0.1 + v / max_val for v in raw]


def moving_average(data, window_size=5):
    """计算滑动平均（不足 window_size 时用当前长度）"""
    result = []
    for i in range(1, len(data) + 1):
        w = min(window_size, i)
        result.append(np.mean(data[i - w:i]))
    return np.array(result)


# ==========================
# 主程序
# ==========================
if __name__ == '__main__':
    fl_system = FederatedLearningWithAttention()
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))

    # ==========================
    # 数据集
    # ==========================
    #train_data, test_loader = utils.data(args)
    #input, output, train_data, val_loader = utils.data_split(
    #    train_data, len(test_loader.dataset), args)

    # ==========================
    # 模型
    # ==========================
    """
    if args.model == 'mlp':
        global_model = models.FC2Layer(input, output)
    elif args.model == 'cnn2':
        global_model = models.CNN2Layer(input, output, args.data)
    elif args.model == 'cnn3':
        global_model = models.CNN3Layer()
    elif args.model == 'LeNet5':
        global_model = models.LeNet5()
    elif args.model == 'lstm':
        global_model = models.LSTMModel(input_size=25, hidden_size=64)
    else:
        global_model = models.Linear(input, output)
"""
    global_model = models.LSTMModel(input_size=25, hidden_size=64)
    textio.cprint(str(summary(global_model)))
    global_model.to(args.device)

    train_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_criterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # 学习曲线记录
    train_loss_list, val_acc_list = [], []

    # ==========================
    # 仅评估模式
    # ==========================
    """
    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model))
        test_acc = utils.test(test_loader, global_model, test_criterion, args.device)
        textio.cprint(f'eval test_acc: {test_acc:.0f}%')
        gc.collect()
        sys.exit()
    """
    # ==========================
    # 中心化训练
    # ==========================
    if not args.federated_learning:
        """
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.train_batch_size, shuffle=True
        )
        optimizer = (
            optim.SGD(global_model.parameters(), lr=args.lr, momentum=args.momentum)
            if args.optimizer == 'sgd'
            else optim.Adam(global_model.parameters(), lr=args.lr)
        )

        for global_epoch in tqdm(range(args.global_epochs)):
            train_loss = utils.train_one_epoch(
                train_loader, global_model, optimizer,
                train_criterion, args.device, args.local_iterations
            )
            val_acc = utils.test(val_loader, global_model, train_criterion, args.device)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)
            print(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | val_acc: {val_acc:.0f}%')

        test_acc = utils.test(test_loader, global_model, test_criterion, args.device)
        print(f'final centralized test_acc: {test_acc:.0f}%')
    """

    # ==========================
    # 联邦学习训练
    # ==========================
    else:
        user_data_len = [16000] * args.num_users
        local_models = federated_utils.federated_setup2(global_model, args)
        mechanism = federated_utils.JoPEQ(args)

        SNR_list, all_users_loss, all_acc = [], {}, {}
        pingjun = np.zeros((args.num_users, 200))
        remind_budget = [600] * args.num_users

        for global_epoch in tqdm(range(args.global_epochs)):
            federated_utils.distribute_model(local_models, global_model)

            users_loss, acc,  MAE = [], [], []
            Distance = np.zeros(args.num_users)
            random.seed(global_epoch)

            epochs = [random.randint(1, 30) if x <= 7 else 30 for x in range(10)]
            print("轮数", epochs)

            for user_idx in range(args.num_users):
                user_loss, model_snapshots = [], []
                user = local_models[user_idx]

                x, m = utils.test2(user_idx, user['model'], test_criterion, args.device, global_epoch)
                acc.append(x);  MAE.append(m)

                for local_epoch in range(epochs[user_idx]):
                    train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],train_criterion, args.device, args.local_iterations, DP=False)
                    if args.lr_scheduler:
                        user['scheduler'].step(train_loss)

                    model_snapshots.append(copy.deepcopy(list(user['model'].parameters())))
                    user_loss.append(train_loss)

                users_loss.append(mean(user_loss))
                pingjun[user_idx, global_epoch] = federated_utils.compute_euclidean_distance(
                    list(global_model.parameters()), list(user['model'].parameters())
                )
                Distance[user_idx] = (
                    (pingjun[user_idx, global_epoch] + pingjun[user_idx, global_epoch-1]) / 2
                    if global_epoch > 0 else pingjun[user_idx, global_epoch]
                )

            all_acc[global_epoch], all_users_loss[global_epoch] = acc, users_loss

            # 用户打分
            if global_epoch > 0:
                a = [(1 - users_loss[i] / all_users_loss[global_epoch - 1][i]) for i in range(args.num_users)]
                b = [(1 - acc[i] / all_acc[global_epoch - 1][i]) for i in range(args.num_users)]
                score = [
                    0.25 * a[i]/max(a) + 0.5 * epochs[i]/max(epochs) + 0.25 * b[i]/max(b)
                    for i in range(args.num_users)
                ]
            else:
                score = epochs

            train_loss = mean(users_loss)
            threshold = 0.8 - 0.15 * (global_epoch // 30)
            combination = [i for i, s in enumerate(score) if s > threshold]

            # 保证至少选出 6 个用户
            if len(combination) <= 6:
                remaining_indices = [i for i, s in enumerate(score) if s <= threshold]
                additional_indices = sorted(
                    remaining_indices, key=lambda i: score[i], reverse=True
                )[:6 - len(combination)]
                combination.extend(additional_indices)
                combination.sort()

            score1 = 1 + (global_epoch // 30) / 3
            score2 = 3 - (global_epoch // 30) / 3

            acc_matrix = np.array([all_acc[e] for e in sorted(all_acc.keys())])
            loss_matrix = np.array([all_users_loss[e] for e in sorted(all_users_loss.keys())])

            move_acc = np.array([moving_average(acc_matrix[:, u], 5) for u in range(acc_matrix.shape[1])]).T
            move_loss = np.array([moving_average(loss_matrix[:, u], 5) for u in range(loss_matrix.shape[1])]).T

            x1 = [v * score1 for v in normalize(move_acc[-1], combination)]
            x2 = [v * score1 for v in normalize(move_loss[-1], combination)]
            x3 = [v * 1 for v in normalize(user_data_len, combination)]
            x4 = [v * score2 for v in normalize(epochs, combination)]

            attention_weights = fl_system.aggregate_models(x1, x2, x3, len(combination), x4)

            # 隐私预算更新
            if global_epoch < 80:
                Remind_budget = [x - 60 * ((99 - global_epoch) // 10) for x in remind_budget]
                eplison = train(Distance, 90 + global_epoch % 10, global_epoch,
                                Remind_budget, users_loss, combination)
            else:
                eplison = train(Distance, global_epoch, global_epoch,
                                remind_budget, users_loss, combination)

            for i in range(args.num_users):
                eplison[i] = max(3, min(eplison[i], 20)) / 10

            SNR = federated_utils.aggregate3(local_models, global_model, combination,
                                             attention_weights, eplison, args)
            SNR_list.append(SNR)

            for i in combination:
                remind_budget[i] -= eplison[i] * 10

            # 结果记录
            val_acc = mean(acc)
            train_loss_list.append(train_loss)
            val_acc_list.append(val_acc)


            boardio.add_scalar('train', train_loss, global_epoch)
            boardio.add_scalar('validation', val_acc, global_epoch)

            gc.collect()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(global_model.state_dict(), path_best_model)

            textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss} | '
                          f'RMSE: {val_acc:.3f} | '
                          f'MAE : {mean(MAE):.3f} | ')

        # 保存曲线与时间
        np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
        np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)

        elapsed_min = (time.time() - start_time) / 60
        textio.cprint(f'total execution time: {elapsed_min:.0f} min')
