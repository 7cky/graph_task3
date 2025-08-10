#没有loss
import pickle
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from data_loader import load_dataset, create_data_loader
from models import GNNModel
from train import train, test
from visualization import plot_results, save_results
from utils import get_device, get_input_dim, get_output_dim
import time
import os
import datetime  # 新增：用于生成时间戳

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='图分类任务')
    parser.add_argument('--dataset', type=str, default='MUTAG', 
                      choices=['MUTAG', 'PROTEINS', 'IMDB-BINARY', 'ZINC'],
                      help='数据集名称')
    parser.add_argument('--model', type=str, default='GCN',
                      choices=['GCN', 'GAT', 'GraphSAGE', 'GIN'],
                      help='GNN模型类型')
    parser.add_argument('--pooling', type=str, default='AvgPooling',
                      choices=['AvgPooling', 'MaxPooling', 'MinPooling'],
                      help='池化方法')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='网络层数')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='批处理大小')
    parser.add_argument('--lr', type=float, default=0.01,
                      help='学习率')
    parser.add_argument('--epochs', type=int, default=100,
                      help='训练轮数')
    parser.add_argument('--results_dir', type=str, default='../results/',
                      help='结果保存目录')
    # 修改：将--exp_name设为可选参数，不再强制要求
    parser.add_argument('--exp_name', type=str, default=None,
                      help='实验名称（不指定则自动生成）')
    args = parser.parse_args()

    # 新增：自动生成实验名称
    if args.exp_name is None:
        # 获取当前时间戳，确保唯一性
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 基于关键参数生成实验名称
        args.exp_name = f"{args.dataset}_{args.model}_{args.pooling}_hd{args.hidden_dim}_layers{args.num_layers}_{timestamp}"

    # 创建实验结果目录
    exp_dir = os.path.join(args.results_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    print(f"实验结果将保存至: {exp_dir}")

    # 加载数据
    print(f"加载数据集: {args.dataset}")
    train_dataset, val_dataset, test_dataset = load_dataset(args.dataset)
    
    # 创建数据加载器
    train_loader = create_data_loader(train_dataset, args.batch_size, shuffle=True)
    val_loader = create_data_loader(val_dataset, args.batch_size, shuffle=False)
    test_loader = create_data_loader(test_dataset, args.batch_size, shuffle=False)
    
    # 获取设备
    device = get_device()
    print(f"使用设备: {device}")
    
    # 获取输入输出维度
    input_dim = get_input_dim(train_dataset)
    output_dim = get_output_dim(train_dataset)
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")
    
    # 初始化模型
    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=output_dim,
        num_layers=args.num_layers,
        model_type=args.model,
        pooling_type=args.pooling
    ).to(device)
    
    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.dataset == 'ZINC':  # ZINC是回归任务
        criterion = torch.nn.MSELoss()
    else:  # 其他是分类任务
        criterion = torch.nn.CrossEntropyLoss()
    
    # 训练模型
    print(f"开始训练 {args.model} 模型，使用 {args.pooling} 池化方法")
    start_time = time.time()
    train_acc_history = []
    val_acc_history = []
    # 初始化loss历史列表
    train_loss_history = []
    val_loss_history = []
    

    
    best_val_acc = 0.0
    best_model = None
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc, _ = train(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = test(model, val_loader, criterion, device)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
    
    total_time = time.time() - start_time
    
    # 加载最佳模型并测试
    model.load_state_dict(best_model)
    test_loss, test_acc, _ = test(model, test_loader, criterion, device)
    print(f"\n测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.4f}")
    print(f"总训练时间: {total_time:.2f} 秒")
    
    # 保存结果
    result = {
        'model': args.model,
        'pooling': args.pooling,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'dataset': args.dataset,
        'train_acc': train_acc,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'total_time': total_time,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'test_loss': test_loss, # 测试集的最终loss
        'exp_name': args.exp_name  # 新增：保存实验名称
    }
    
    
    # 保存所有结果 - 修改这部分代码以处理空文件
    results_path = os.path.join(args.results_dir, 'all_results.pkl')
    if os.path.exists(results_path):
        # 检查文件大小，避免空文件
        if os.path.getsize(results_path) > 0:
            try:
                with open(results_path, 'rb') as f:
                    all_results = pickle.load(f)
                all_results.append(result)
            except (EOFError, pickle.UnpicklingError):
                # 处理文件损坏的情况
                print(f"警告: {results_path} 文件损坏或不完整，将创建新文件")
                all_results = [result]
        else:
            # 处理空文件情况
            print(f"警告: {results_path} 是空文件，将创建新文件")
            all_results = [result]
    else:
        all_results = [result]
    
    # 保存所有结果
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    
    # 可视化结果 - 自动使用当前实验名称作为前缀
    plot_results(all_results, exp_dir)
    save_results(all_results, exp_dir)

    # 新增：保存当前实验的参数配置
    with open(os.path.join(exp_dir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    print(f"实验配置已保存至: {os.path.join(exp_dir, 'config.txt')}")

if __name__ == "__main__":
    main()

    
    

    


