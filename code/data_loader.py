import torch_geometric
from torch_geometric.datasets import TUDataset, ZINC
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import torch

def load_dataset(dataset_name, data_dir='../data/'):
    """加载指定的数据集"""
    if dataset_name in ['MUTAG', 'PROTEINS', 'IMDB-BINARY']:
        dataset = TUDataset(root=data_dir, name=dataset_name, transform=T.ToUndirected())
    elif dataset_name == 'ZINC':
        dataset = ZINC(root=data_dir, subset=True)  # subset为True表示使用小数据集
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")
    
    # 划分训练集、验证集、测试集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset, test_dataset

def create_data_loader(dataset, batch_size=32, shuffle=True):
    """创建数据加载器"""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)