# @Time    : 2020/8/25 
# @Author  : LeronQ
# @github  : https://github.com/LeronQ


# gcnnet.py

import torch
import torch.nn as nn


class GCN(nn.Module):  # GCN模型，向空域的第一个图卷积
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()  # 表示继承父类的所有属性和方法
        self.linear_1 = nn.Linear(in_c, hid_c)  # 定义一个线性层
        self.linear_2 = nn.Linear(hid_c, out_c)  # 定义一个线性层
        self.act = nn.ReLU()  # 定义激活函数

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N] 邻接矩阵，并且将数据送入设备
        graph_data = GCN.process_graph(graph_data)  # 变换邻接矩阵 \hat A = D_{-1/2}*A*D_{-1/2}

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]  流量数据

        B, N = flow_x.size(0), flow_x.size(1)  # batch_size、节点数

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D] H = 6, D = 1把最后两维缩减到一起了，这个就是把历史时间的特征放一起

        # 第一个图卷积层
        output_1 = self.linear_1(flow_x)  # [B, N, hid_C],这个就是 WX，其中W是可学习的参数，X是输入的流量数据（就是flow_x）
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [B, N, N] ,[B, N, hid_c]，就是 \hat AWX

        # 第二个图卷积层
        output_2 = self.linear_2(output_1)# WX
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C] , 就是 \hat AWX

        return output_2.unsqueeze(2)  # 第２维的维度扩张


    @staticmethod
    def process_graph(graph_data):  # 这个就是在原始的邻接矩阵之上，再次变换，也就是\hat A = D_{-1/2}*A*D_{-1/2}
        N = graph_data.size(0) # 获得节点的个数
        matrix_i = torch.eye(N, dtype=torch.float, device=graph_data.device)  # 定义[N, N]的单位矩阵
        graph_data += matrix_i  # [N, N]  ,就是 A+I

        degree_matrix = torch.sum(graph_data, dim=1, keepdim=False)  # [N],计算度矩阵，塌陷成向量，其实就是将上面的A+I每行相加
        degree_matrix = degree_matrix.pow(-1)  # 计算度矩阵的逆，若为0，-1次方可能计算结果为无穷大的数
        degree_matrix[degree_matrix == float("inf")] = 0.  # 让无穷大的数为0

        degree_matrix = torch.diag(degree_matrix)  # 转换成对角矩阵

        return torch.mm(degree_matrix, graph_data)  # 返回 \hat A=D^(-1) * A ,这个等价于\hat A = D_{-1/2}*A*D_{-1/2}
