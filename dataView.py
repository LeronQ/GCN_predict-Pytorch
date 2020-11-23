# @Time    : 2020/8/25
# @Author  : LeronQ
# @github  : https://github.com/LeronQ

import numpy as np
import matplotlib.pyplot as plt


def get_flow(file_name):  # 将读取文件写成一个函数

    flow_data = np.load(file_name)  # 载入交通流量数据
    print([key for key in flow_data.keys()])  # 打印看看key是什么

    print('before flow_data',flow_data["data"].shape)  # (16992, 307, 3)，16992是时间(59*24*12)，307是节点数，3表示每一维特征的维度（类似于二维的列）
    # flow_data = flow_data['data']  # [T, N, D]，T为时间，N为节点数，D为节点特征
    # print('Before flow_data',flow_data.shape)

    flow_data = flow_data['data'].transpose([1, 0, 2])[:,:,0][:,:,np.newaxis]
    return flow_data

def pre_process_data(data, norm_dim):  # 预处理,归一化
    """
    :param data: np.array,原始的交通流量数据
    :param norm_dim: int,归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
    :return:
        norm_base: list, [max_data, min_data], 这个是归一化的基.
        norm_data: np.array, normalized traffic data.
    """
    norm_base = normalize_base(data, norm_dim)  # 计算 normalize base
    norm_data = normalize_data(norm_base[0], norm_base[1], data)  # 归一化后的流量数据

    return norm_base, norm_data  # 返回基是为了恢复数据做准备的


def normalize_base(data, norm_dim):#计算归一化的基
    """
    :param data: np.array, 原始的交通流量数据
    :param norm_dim: int, normalization dimension.归一化的维度，就是说在哪个维度上归一化,这里是在dim=1时间维度上
    :return:
        max_data: np.array
        min_data: np.array
    """
    max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D], keepdims=True就保持了纬度一致
    min_data = np.min(data, norm_dim, keepdims=True)

    return max_data, min_data   # 返回最大值和最小值


def normalize_data(max_data, min_data, data):#计算归一化的流量数据，用的是最大值最小值归一化法
    """
    :param max_data: np.array, max data.
    :param min_data: np.array, min data.
    :param data: np.array, original traffic data without normalization.
    :return:
        np.array, normalized traffic data.
    """
    mid = min_data
    base = max_data - min_data
    normalized_data = (data - mid) / base

    return normalized_data

def slice_data(data, history_length, index, train_mode): #根据历史长度,下标来划分数据样本
    """
    :param data: np.array, normalized traffic data.
    :param history_length: int, length of history data to be used.
    :param index: int, index on temporal axis.
    :param train_mode: str, ["train", "test"].
    :return:
        data_x: np.array, [N, H, D].
        data_y: np.array [N, D].
    """
    if train_mode == "train":
        start_index = index #开始下标就是时间下标本身，这个是闭区间
        end_index = index + history_length #结束下标,这个是开区间
    elif train_mode == "test":
        start_index = index - history_length #　开始下标，这个最后面贴图了，可以帮助理解
        end_index = index # 结束下标
    else:
        raise ValueError("train model {} is not defined".format(train_mode))

    print('data',data.shape)
    data_x = data[:, start_index: end_index]  # 在切第二维，不包括end_index
    data_y = data[:, end_index]  # 把上面的end_index取上

    return data_x, data_y

# 做工程、项目等第一步对拿来的数据进行可视化的直观分析
if __name__ == "__main__":
    traffic_data = get_flow("PeMS_04/PeMS04.npz")

    norm_base, norm_data = pre_process_data(traffic_data,1)
    data_x, data_y = slice_data(norm_data,5,100,'train')

    # print('norm_data',norm_data.shape)
    # print('data_x',data_x.shape)
    # print('data_y',data_y.shape)
    # print(norm_data)