import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns


# 设置中文字体支持，尝试多种常见中文字体
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC", "Arial Unicode MS"]
# 解决负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False
def plot_results(results, save_path):
    """可视化不同模型和池化方法的性能对比"""
    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 转换结果为DataFrame
    df = pd.DataFrame(results)
    
    # 获取最新实验的关键参数，用于命名文件
    latest_exp = results[-1]
    exp_suffix = f"{latest_exp['dataset']}_{latest_exp['model']}_{latest_exp['pooling']}"
    
    # 1. 不同模型的准确率对比 - 文件名包含实验参数
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='test_acc', hue='pooling', data=df)
    plt.title('Comparison')
    plt.xlabel('model')
    plt.ylabel('test accuracy')
    plt.savefig(os.path.join(save_path, f'model_pooling_acc_{exp_suffix}.png'))
    plt.close()
    
    # 2. 训练过程中的准确率变化
    plt.figure(figsize=(12, 6))
    for idx, result in enumerate(results):
        model_name = f"{result['model']}_{result['pooling']}"
        plt.plot(result['train_acc_history'], label=f'{model_name} (训练)')
        plt.plot(result['val_acc_history'], label=f'{model_name} (验证)')
    
    # plt.title('训练过程中的准确率变化')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'training_acc_history_{exp_suffix}.png'))
    plt.close()
    
    # 3. 不同模型的运行时间对比
    plt.figure(figsize=(12, 6))
    sns.barplot(x='model', y='total_time', hue='pooling', data=df)
    plt.title('Comparison')
    plt.xlabel('model')
    plt.ylabel('Time')
    plt.savefig(os.path.join(save_path, f'model_pooling_time_{exp_suffix}.png'))
    plt.close()

def save_results(results, save_path):
    """保存结果为CSV文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df = pd.DataFrame(results)
    
    # 文件名包含实验参数
    latest_exp = results[-1]
    exp_suffix = f"{latest_exp['dataset']}_{latest_exp['model']}_{latest_exp['pooling']}"
    csv_path = os.path.join(save_path, f'results_{exp_suffix}.csv')
    
    df.to_csv(csv_path, index=False)
    print(f"结果已保存至 {csv_path}")




# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import seaborn as sns

# # 修复负号显示问题
# plt.rcParams["axes.unicode_minus"] = False

# def plot_results(results, save_path):
#     """Visualization including accuracy, loss, and running time"""
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     df = pd.DataFrame(results)
#     latest_exp = results[-1]
#     exp_suffix = f"{latest_exp['dataset']}_{latest_exp['model']}_{latest_exp['pooling']}"
    
#     # 1. 测试准确率对比
#     plt.figure(figsize=(12, 6))
#     sns.barplot(x='model', y='test_acc', hue='pooling', data=df)
#     plt.title('Test Accuracy Comparison: Models vs Pooling Methods')
#     plt.xlabel('Model Type')
#     plt.ylabel('Test Accuracy')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig(os.path.join(save_path, f'model_pooling_acc_{exp_suffix}.png'), bbox_inches='tight')
#     plt.close()
    
#     # 2. 训练过程中的准确率曲线
#     plt.figure(figsize=(12, 6))
#     for _, result in enumerate(results):
#         model_name = f"{result['model']}_{result['pooling']}"
#         plt.plot(result['train_acc_history'], label=f'{model_name} (Train)', linewidth=2)
#         plt.plot(result['val_acc_history'], label=f'{model_name} (Validation)', linestyle='--', linewidth=2)
#     plt.title('Accuracy Trends During Training')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.grid(linestyle='--', alpha=0.7)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f'training_acc_history_{exp_suffix}.png'), bbox_inches='tight')
#     plt.close()
    
#     # 3. 训练过程中的损失曲线（新增）
#     plt.figure(figsize=(12, 6))
#     for _, result in enumerate(results):
#         model_name = f"{result['model']}_{result['pooling']}"
#         plt.plot(result['train_loss_history'], label=f'{model_name} (Train)', linewidth=2)
#         plt.plot(result['val_loss_history'], label=f'{model_name} (Validation)', linestyle='--', linewidth=2)
#     plt.title('Loss Trends During Training')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.grid(linestyle='--', alpha=0.7)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f'training_loss_history_{exp_suffix}.png'), bbox_inches='tight')
#     plt.close()
    
#     # 4. 测试损失对比（新增）
#     plt.figure(figsize=(12, 6))
#     sns.barplot(x='model', y='test_loss', hue='pooling', data=df)
#     plt.title('Test Loss Comparison: Models vs Pooling Methods')
#     plt.xlabel('Model Type')
#     plt.ylabel('Test Loss')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig(os.path.join(save_path, f'model_pooling_loss_{exp_suffix}.png'), bbox_inches='tight')
#     plt.close()
    
#     # 5. 运行时间对比
#     plt.figure(figsize=(12, 6))
#     sns.barplot(x='model', y='total_time', hue='pooling', data=df)
#     plt.title('Total Running Time Comparison')
#     plt.xlabel('Model Type')
#     plt.ylabel('Total Time (seconds)')
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.savefig(os.path.join(save_path, f'model_pooling_time_{exp_suffix}.png'), bbox_inches='tight')
#     plt.close()

# def save_results(results, save_path):
#     """Save all metrics including loss"""
#     os.makedirs(os.path.dirname(save_path), exist_ok=True)
#     df = pd.DataFrame(results)
#     latest_exp = results[-1]
#     exp_suffix = f"{latest_exp['dataset']}_{latest_exp['model']}_{latest_exp['pooling']}"
#     csv_path = os.path.join(save_path, f'results_{exp_suffix}.csv')
#     df.to_csv(csv_path, index=False)
#     print(f"Results (including loss) saved to: {csv_path}")