import torch
import torch.nn.functional as F
from tqdm import tqdm
import time

def train(model, train_loader, optimizer, criterion, device):
    """训练模型"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for data in tqdm(train_loader, desc="Training"):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        
        # 根据数据集类型选择合适的损失函数
        if data.y.dim() == 2 and data.y.size(1) > 1:
            # 多标签分类
            loss = criterion(out, data.y.float())
            pred = (out > 0).float()
            correct += int((pred == data.y).sum())
            total += data.y.numel()
        else:
            # 单标签分类
            loss = criterion(out, data.y.view(-1))
            pred = out.argmax(dim=1)
            correct += int((pred == data.y.view(-1)).sum())
            total += data.num_graphs
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_time = time.time() - start_time
    train_acc = correct / total if total > 0 else 0
    
    return total_loss / len(train_loader), train_acc, train_time

def test(model, test_loader, criterion, device):
    """测试模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            
            # 根据数据集类型选择合适的损失函数
            if data.y.dim() == 2 and data.y.size(1) > 1:
                # 多标签分类
                loss = criterion(out, data.y.float())
                pred = (out > 0).float()
                correct += int((pred == data.y).sum())
                total += data.y.numel()
            else:
                # 单标签分类
                loss = criterion(out, data.y.view(-1))
                pred = out.argmax(dim=1)
                correct += int((pred == data.y.view(-1)).sum())
                total += data.num_graphs
            
            total_loss += loss.item()
    
    test_time = time.time() - start_time
    test_acc = correct / total if total > 0 else 0
    
    return total_loss / len(test_loader), test_acc, test_time