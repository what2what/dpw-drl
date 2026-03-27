import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear
from torch.distributions import Categorical
from itertools import accumulate
from model.gnn import GNN
torch.set_printoptions(precision=10)

class REINFORCE(nn.Module):
    def __init__(self, args):
        super(REINFORCE, self).__init__()
        self.args = args
        self.policy_num_layers = args.policy_num_layers
        self.hidden_dim = args.hidden_dim
        self.gnn = GNN(args) # 实例化 GNN 模块，用于提取节点（工序和机器）的特征
        # 构建策略网络 (Policy Network) - 一个简单的 MLP
        self.layers = torch.nn.ModuleList() # policy network
        # 输入维度解释：
        # self.hidden_dim * 2：因为决策需要考虑 "当前工序特征" + "目标机器特征"
        # + 1：额外的 1 维是 "归一化的加工时间"
        self.layers.append(nn.Linear(self.hidden_dim * 2 + 1, self.hidden_dim))
        for _ in range(self.policy_num_layers - 2):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        # 输出层：输出 1 个标量分数
        self.layers.append(nn.Linear(self.hidden_dim, 1))

        # 存储强化学习训练所需的轨迹数据
        self.log_probs = []# 记录动作的对数概率
        self.entropies = []# 记录熵（用于鼓励探索）
        self.rewards = []  # 记录每一步的奖励
        self.baselines = []# 记录基线（用于计算优势函数，减少方差）
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        
    def forward(self, avai_ops, data, op_unfinished, max_process_time, greedy=False):
        # 1. 使用 GNN 获取当前所有工序和机器的嵌入向量 (Embedding)
        x_dict = self.gnn(data)

        # 准备容器存储所有候选动作的特征
        score = torch.empty(size=(0, self.args.hidden_dim * 2 + 1)).to(self.args.device)

        for op_info in avai_ops:
            normalize_process_time = torch.tensor([op_info['process_time'] / max_process_time], dtype=torch.float32, device=self.args.device)
            # 特征拼接核心逻辑：
            # [机器特征, 工序特征, 加工时间]
            # 维度变化：hidden_dim + hidden_dim + 1
            score = torch.cat((score, torch.cat((x_dict['m'][op_info['m_id']], x_dict['op'][op_unfinished.index(op_info['node_id'])], normalize_process_time), dim=0).unsqueeze(0)), dim=0)

        # 3. 将特征输入 MLP，计算每个候选动作的评分 (Score)
        for i in range(self.policy_num_layers - 1):
            score = F.leaky_relu(self.layers[i](score))
        # 最后一层输出 raw score
        score = self.layers[self.policy_num_layers - 1](score)

        # 4. 计算概率分布
        probs = F.softmax(score, dim=0).flatten()
        # 创建分类分布
        dist = Categorical(probs)
        # 5. 采样动作 (Action Sampling)
        if greedy == True:
            # 贪婪模式（通常用于测试/验证）：直接选概率最大的
            idx = torch.argmax(score)
        else:
            # 训练模式：根据概率随机采样（保留探索性）
            idx = dist.sample()

        self.log_probs.append(dist.log_prob(idx))
        self.entropies.append(dist.entropy())
        return idx.item(), probs[idx].item()
    
    def calculate_loss(self, device):
        loss = []
        # 1. 计算回报 (Returns/Gt)
        # accumulate 相当于计算后缀和。这里是在计算从当前步开始的累计奖励。
        # [::-1] 的两次使用是为了从后往前计算累计和（Cost-to-Go）。
        returns = torch.FloatTensor(list(accumulate(self.rewards[::-1]))[::-1]).to(device)
        policy_loss = 0.0
        entropy_loss = 0.0

        for log_prob, entropy, R, baseline in zip(self.log_probs, self.entropies, returns, self.baselines):
            # 2. 计算优势函数 (Advantage)
            # 这是一个最小化问题（如最小化完工时间）。
            if baseline == 0:
                advantage = R * -1
            else:
                # (实际回报 - 基线) / 基线。
                # 乘以 -1 是因为通常 RL 是最大化奖励，而调度问题通常是最小化时间。
                # 如果 R (耗时) > baseline (平均耗时)，结果为正，乘 -1 变负 -> 抑制该动作。
                # 如果 R (耗时) < baseline (平均耗时)，结果为负，乘 -1 变正 -> 鼓励该动作。
                advantage = ((R - baseline) / baseline) * -1
            # 3. 最终 Loss 公式
            # Loss = - (log_prob * advantage) - (系数 * entropy)
            # 第一项是 Policy Gradient：最大化优势大的动作的概率。
            # 第二项是 熵正则化：鼓励策略保持随机性，防止过早收敛到局部最优。
            loss.append(-log_prob * advantage - self.args.entropy_coef * entropy)
            policy_loss += log_prob * advantage
            entropy_loss += entropy
        # 返回平均 Loss 用于反向传播
        return torch.stack(loss).mean(), policy_loss / len(self.log_probs), entropy_loss / len(self.log_probs)
 
    def clear_memory(self):
        del self.log_probs[:]
        del self.entropies[:]
        del self.rewards[:]
        del self.baselines[:]
        return
         