import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_distance(x, y):
    # x and y are expected to be tensors of shape (batch_size, features)
    dot_product = torch.sum(x * y, dim=-1)
    norm_v1 = torch.norm(x, dim=-1)
    norm_v2 = torch.norm(y, dim=-1)
    cosine_sim = dot_product / (norm_v1 * norm_v2)
    return torch.stack([torch.norm(cosine_sim)])


class AttentionMechanism(nn.Module):
    def __init__(self):
        super(AttentionMechanism, self).__init__()

    def forward(self, K, V):
        # K: local features of each agent (rewards, losses, model similarity)
        # V: global features of all agents (max reward, min loss, max model similarity)
        # Calculate attention scores (dot product of K and V)
        norm_K = torch.norm(K, dim=-1, keepdim=True)
        norm_V = torch.norm(V, dim=-1, keepdim=True)
        cosine_similarity = torch.sum(K * V, dim=-1) / (norm_K[:, 0] * norm_V[0])
        attention_weights = F.softmax(cosine_similarity, dim=-1)
        # Calculate weighted K based on attention weights
        #weighted_K = torch.sum(attention_weights * K, dim=0)

        return attention_weights


def compute_agent_features(loss, rewards_base, combination_data_len, len, step):
    loss = torch.tensor([loss[k] for k in range(len)], dtype=torch.float32)
    rewards_bases = torch.tensor([rewards_base[k] for k in range(len)], dtype=torch.float32)
    batch_size = torch.tensor([combination_data_len[k] for k in range(len)], dtype=torch.float32)
    epoch = torch.tensor([step[k] for k in range(len)], dtype=torch.float32)
    # Combine rewards, losses, and similarity into matrix K (features of each agent)
    K = torch.stack([loss, rewards_bases, batch_size, epoch], dim=1)
    return K


def compute_global_features(K):
    rewards = K[:, 0]
    rewards_base = K[:, 1]
    batch_size = K[:, 2]
    epochs = K[:, 3]
    # Calculate global features V
    min_rewards = torch.min(rewards)
    max_rewards_base = torch.min(rewards_base)
    max_batch_size = torch.max(batch_size)
    max_epoch = torch.max(epochs)
    Q = torch.stack([min_rewards, max_rewards_base, max_batch_size, max_epoch], dim=0)
    return Q


class FederatedLearningWithAttention:
    def __init__(self):
        self.attention_layer = AttentionMechanism()

    def aggregate_models(self, Loss, rewards_base, combination_data_len, len, step):
        # Compute local features K for each agent
        K = compute_agent_features(Loss, rewards_base, combination_data_len, len, step)
        # Compute global features V
        Q = compute_global_features(K)
        # Use attention mechanism to calculate weighted local features K
        attention_weights = self.attention_layer(K, Q)

        # Use weighted K for model aggregation
        return attention_weights