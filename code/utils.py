import torch

def get_device():
    """获取可用的设备（GPU或CPU）"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_input_dim(dataset):
    """获取数据集的输入维度"""
    return dataset[0].x.size(1)

def get_output_dim(dataset):
    """获取数据集的输出维度（类别数）"""
    if dataset[0].y.dim() == 2:
        return dataset[0].y.size(1)
    else:
        # 对于单标签分类，找到最大的类别索引并加1
        ys = [data.y.item() for data in dataset]
        return max(ys) + 1 if ys else 1