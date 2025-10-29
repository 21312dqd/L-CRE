# LCRE - 基因表达分类器

基于深度学习的植物基因表达水平预测工具，专注于从启动子和终止子序列预测基因表达模式。

## 📋 项目简介

LCRE (Low/High gene expression Classification using Regulatory Elements) 是一个利用卷积神经网络和LSTM的深度学习框架，通过分析基因的启动子和终止子序列来预测基因表达水平。本项目当前实现针对番茄（*Solanum lycopersicum*）基因组的分类器。

## ✨ 主要特性

- **序列编码**: DNA序列的one-hot编码
- **深度学习架构**: 结合CNN和LSTM的混合神经网络
- **染色体级别验证**: 使用留一染色体交叉验证策略
- **类别平衡**: 自动平衡训练数据中的高/低表达基因
- **性能追踪**: 每个epoch保存模型并记录性能指标

## 🔧 环境依赖

### Python版本
- Python 3.7+

### 核心依赖包
```
tensorflow >= 2.x
pandas
numpy
pyranges
pyfaidx
scikit-learn
```

### 安装依赖
```bash
pip install tensorflow pandas numpy pyranges pyfaidx scikit-learn
```

## 📁 数据要求

项目需要以下数据文件结构：

```
project_root/
├── tpm_counts/
│   └── solanum_counts.csv          # TPM表达量数据
├── gene_models/
│   └── Solanum_lycopersicum.SL3.0.52.gtf  # 基因注释文件
├── genomes/
│   └── Solanum_lycopersicum.SL3.0.dna.toplevel.fa  # 参考基因组
├── validation_genes.pickle          # 验证基因ID集合
└── saved_models/                    # 模型保存目录（自动创建）
```

### 数据格式说明

1. **TPM数据** (`solanum_counts.csv`):
   - 必须包含 `logMaxTPM` 列
   - 行索引为基因ID

2. **GTF文件**: 标准基因组注释格式，需包含：
   - 基因坐标信息
   - `gene_biotype` 字段（仅使用protein_coding基因）

3. **FASTA文件**: 标准基因组序列格式

4. **验证基因pickle文件**: 包含用于验证的基因ID字典

## 🚀 使用方法

### 基本用法

```bash
python lcre_classifier.py
```

### 主要流程

1. **数据加载与预处理**
   - 读取TPM数据并按25%和75%分位数分类
   - 0: 低表达基因 (≤ 25th percentile)
   - 1: 高表达基因 (≥ 75th percentile)
   - 2: 中等表达基因 (不用于训练)

2. **序列提取**
   - 启动子区域: TSS上游1000bp + 下游500bp
   - 终止子区域: TES上游500bp + 下游1000bp
   - 20bp零填充分隔启动子和终止子序列

3. **模型训练**
   - 对每条染色体进行独立验证
   - 训练数据: 其余11条染色体
   - 验证数据: 当前染色体
   - 35个训练周期，批次大小64

## 🏗️ 模型架构

```
输入层: (序列长度, 4) - One-hot编码的DNA序列
↓
Conv1D(64, kernel=8) + ReLU
Conv1D(64, kernel=8) + ReLU
MaxPooling1D(8)
Dropout(0.25)
LayerNormalization
↓
LSTM(64, return_sequences=True)
Dropout(0.25)
↓
Conv1D(128, kernel=8) + ReLU
Conv1D(128, kernel=8) + ReLU
MaxPooling1D(8)
Dropout(0.25)
↓
Conv1D(64, kernel=8) + ReLU
Conv1D(64, kernel=8) + ReLU
MaxPooling1D(8)
Dropout(0.25)
↓
Flatten
Dense(128) + ReLU
Dropout(0.25)
Dense(64) + ReLU
Dense(1) + Sigmoid
↓
输出: 基因表达水平概率 (0-1)
```

### 模型特点
- **损失函数**: Binary crossentropy
- **优化器**: Adam (学习率: 0.0001)
- **正则化**: Dropout layers (0.25) + Layer Normalization
- **序列长度**: 3020bp (1500 + 20 + 1500)

## 📊 输出结果

### 保存的模型文件
```
saved_models/
└── solanum_model_{chromosome}/
    ├── best_model.h5              # 最佳验证性能模型
    ├── performance_log.csv        # 训练过程记录
    └── epoch_models/
        ├── model_epoch_0.h5
        ├── model_epoch_1.h5
        └── ...
```

### 结果文件
- `../results/sol_root_result.csv`: 包含所有染色体的性能指标
  - accuracy: 验证准确率
  - auROC: 验证集ROC曲线下面积
  - organism: 物种标识
  - training_size: 训练样本数量

## 📈 性能评估

模型使用以下指标评估：
- **准确率 (Accuracy)**: 正确分类的比例
- **auROC**: ROC曲线下面积，评估分类器整体性能

## 🔧 配置参数

可在 `CONFIG` 字典中修改的参数：

```python
CONFIG = {
    'MAPPED_READS': 'solanum_counts.csv',     # TPM数据文件
    'GENE_MODEL': 'Solanum_lycopersicum.SL3.0.52.gtf',  # GTF文件
    'GENOME': 'Solanum_lycopersicum.SL3.0.dna.toplevel.fa',  # FASTA文件
    'PICKLE_KEY': 'sol',                       # 验证基因pickle键
    'NUM_CHROMOSOMES': 12                      # 染色体数量
}
```

其他可调整的参数：
- `upstream`: 上游延伸碱基数 (默认: 1000)
- `downstream`: 下游延伸碱基数 (默认: 500)
- `batch_size`: 批次大小 (默认: 64)
- `epochs`: 训练周期数 (默认: 35)
- `learning_rate`: 学习率 (默认: 0.0001)

## 🧬 核心类和函数

### `encode_sequence(sequence)`
将DNA序列转换为one-hot编码数组。

### `SequenceLoader`
从FASTA和GTF文件加载和处理基因组序列。

### `ConvolutionalNetwork`
构建、训练和评估卷积神经网络模型。

### `prepare_validation_sequences()`
准备带有标签和基因ID的验证序列。

### `train_tomato_classifier()`
主训练函数，协调整个训练流程。

## 💡 使用建议

1. **GPU加速**: 代码自动配置GPU内存增长，建议使用GPU训练
2. **数据平衡**: 自动平衡低/高表达基因数量，确保无偏训练
3. **序列掩码**: 在TSS和TES位置设置掩码，避免模型学习位置偏差
4. **检查点**: 每个epoch保存模型，可根据需要恢复训练

## 🐛 故障排除

### 常见问题

1. **内存不足**: 减小batch_size或使用更小的序列长度
2. **数据文件未找到**: 检查文件路径和目录结构
3. **GPU不可用**: 代码会自动回退到CPU训练

## 📝 引用

如果您使用了本工具，请引用相关研究。

## 📄 许可证

请根据您的项目需求添加适当的许可证。

## 👥 贡献

欢迎提交问题和改进建议！

## 📧 联系方式

如有问题，请通过 [您的联系方式] 联系。

---

**注意**: 本工具专门设计用于植物基因组分析，特别是番茄基因组。如需应用于其他物种，需要调整配置参数和数据路径。
