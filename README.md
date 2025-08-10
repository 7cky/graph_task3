# Graph_task3 图分类

## 一、实验目的
测试不同图神经网络模型（GCN、GAT、GraphSAGE、GIN）在MUTAG和PROTEINS数据集上的性能，并分析AvgPooling、MaxPooling、MinPooling三种池化方法对模型效果的影响

## 二、实验过程
1. 图分类
根据图的整体结构和节点属性对图进行类别划分。

2. 数据集
本实验选用TUDataset中的MUTAG和PROTEINS数据集，ZINC数据集需要访问外网下载。
   - a) MUTAG数据集
   生物化学领域数据集，包含硝基芳香族化合物的分子结构数据，共188个图，每个图代表一个分子，节点表示原子，边表示原子间的化学键。节点特征包含原子类型等化学属性，边特征为化学键类型。主要为二分类任务，判断分子是否具有致突变性，标签为0（非致突变）和1（致突变）。
   - b) PROTEINS数据集
   蛋白质结构领域数据集，每个图对应蛋白质的一个结构域，共1113个图。节点表示氨基酸残基，边表示残基间的空间距离。节点特征包含氨基酸的物理化学属性。也是二分类任务，判断蛋白质结构域是否为酶（enzymatic），标签为0和1。

3. 池化
用于将图中所有节点的特征聚合为一个全局图特征，以便进行最终的分类判断。
   - a) 平均池化（AvgPooling）
   对图中所有节点的特征进行元素级平均，得到图的全局特征向量。考虑所有节点的贡献，对异常值（如噪声节点）较稳健，但可能弱化重要节点的特征。
   - b) 最大池化（MaxPooling）
   对图中所有节点的特征进行元素级最大值提取，保留每个特征维度的最大值作为图特征。突出图中具有极端特征的节点（如关键功能节点），对局部显著特征敏感，但易受噪声影响。
   - c) 最小池化（MinPooling）
   对图中所有节点的特征进行元素级最小值提取，保留每个特征维度的最小值作为图特征。

## 三、实验结果
1. 运行命令及超参数设置
```
python code/main.py --dataset MUTAG --model GCN --pooling AvgPooling
```
   - --dataset: 数据集名称，可选 MUTAG, PROTEINS, 
   - --model: GNN 模型类型，可选 GCN, GAT, GraphSAGE, GIN
   - --pooling: 池化方法，可选 AvgPooling, MaxPooling, MinPooling
   - --hidden_dim: 隐藏层维度，设为64
   - --num_layers: 网络层数，设为2
   - --batch_size: 批处理大小，设为32
   - --lr: 学习率，0.01
   - --epochs: 训练轮数，100

2. MUTAG
   <img width="1200" height="600" alt="model_pooling_time_MUTAG_GIN_MaxPooling" src="https://github.com/user-attachments/assets/d48828f1-f92f-454d-9388-5083280ff021" />
   <img width="1200" height="600" alt="model_pooling_acc_MUTAG_GIN_MaxPooling" src="https://github.com/user-attachments/assets/c992e7ef-d0b2-4218-ad62-03ce6692a7f7" />
3.PROTEINS
  <img width="1200" height="600" alt="model_pooling_time_PROTEINS_GraphSAGE_MinPooling (1)" src="https://github.com/user-attachments/assets/e95edbb2-4c67-41f1-a357-7caac85f4742" />
  <img width="1200" height="600" alt="model_pooling_acc_PROTEINS_GraphSAGE_MinPooling" src="https://github.com/user-attachments/assets/ac98aa66-018f-466e-b047-32ef6dfa1c3a" />

