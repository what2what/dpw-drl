import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HeteroConv, Linear, GINEConv, MLP
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor

class GINEConv(MessagePassing):
    def __init__(self, nn, eps = 0, train_eps = False, edge_dim = None, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index, edge_attr = None, size = None):
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if out.shape[0] == 1:
            out = torch.cat((out, out), dim=0)
            return self.nn(out)[:1]

        return self.nn(out)

    def message(self, x_j, edge_attr = None):
        if edge_attr is not None:
            return torch.cat((x_j, edge_attr), dim=1)
        else:
            return x_j

    def __repr__(self):
        return f'{self.__class__.__name__}(nn={self.nn})'

class GNN(nn.Module):
    def __init__(self, args):
        super(GNN, self).__init__()
        self.args = args
        self.num_layers = args.GNN_num_layers # GNN 层数
        self.hidden_dim = args.hidden_dim     # 隐藏层维度
        self.convs = torch.nn.ModuleList()  # 存储每一层的卷积列表

        if args.delete_node == True:
            # 如果删除某些节点信息，输入4维转5维
            self.m_trans_fc = Linear(4, 5)
            in_dim = 5
        else:
            # 否则输入4维转7维
            self.m_trans_fc = Linear(4, 7)
            in_dim = 7
        
        for _ in range(self.num_layers):
            nn1 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn2 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn3 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            nn4 = MLP([in_dim, self.hidden_dim, self.hidden_dim])
            # HeteroConv 允许对不同的边类型使用不同的卷积算子
            # 这里的字典键是元组：(源节点类型, 关系类型, 目标节点类型)
            conv = HeteroConv({
                ('op', 'to', 'op'): GINEConv(nn=nn1), # 工序到工序
                ('op', 'to', 'm'): GINEConv(nn=nn2),  # 工序到机器
                ('m', 'to', 'op'): GINEConv(nn=nn3),  # 机器到工序
                ('m', 'to', 'm'): GINEConv(nn=nn4),   # 机器到机器
                            }, aggr='sum')  # 不同关系聚合来的结果相加
            self.convs.append(conv) #将每一层的mlp加入
            # 下一层的输入维度等于当前层的输出维度 (hidden_dim)
            in_dim = self.hidden_dim
        # 最后的输出线性层
        self.op_fc = Linear(self.hidden_dim, self.hidden_dim)
        self.m_fc = Linear(self.hidden_dim, self.hidden_dim)

    
    def forward(self, data):
        # data 是 PyG 的 HeteroData 对象
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # 1. 特征预处理
        # 对 'm' (机器) 类型的节点特征进行线性变换
        x_dict['m'] = self.m_trans_fc(x_dict['m'])

        for conv in self.convs:
            # 执行异构卷积
            x_dict = conv(x_dict, edge_index_dict)
            # 激活函数 LeakyReLU
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        # 3. 输出层变换
        x_dict['op'] = self.op_fc(x_dict['op'])
        x_dict['m'] = self.m_fc(x_dict['m'])
        
        return x_dict