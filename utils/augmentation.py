import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


def augment_minority_knn(X, y, n_samples=1000, k=5):
    """
    仅对少数类做KNN插值生成合成样本
    
    Args:
        X (array-like, shape (n_samples, n_features)): 特征数据
        y (array-like, shape (n_samples,)): 标签数据
    """
    # 确定少数类标签：比较两类样本数量，数量少的为少数类
    # 假设标签只有0和1两种取值
    minority_label = 1 if np.sum(y == 1) < np.sum(y == 0) else 0
    
    # 提取少数类的特征数据
    X_min = X[y == minority_label]
    
    # 如果少数类样本数小于2，无法生成合成样本，直接返回原始数据
    if len(X_min) < 2:
        return X, y

    # 确定实际使用的近邻数：不超过k，且不超过少数类样本数-1（排除自身）
    n_neighbors = min(k, len(X_min) - 1)
    
    # 构建KNN模型，拟合少数类数据
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(X_min)
    
    # 用于存储生成的合成样本
    synthetic = []
    
    # 生成指定数量的合成样本
    for _ in range(n_samples):
        # 随机选择一个少数类样本作为基准样本
        idx = np.random.randint(0, len(X_min))
        
        # 找到该样本的k个近邻的索引
        # kneighbors返回距离和索引，这里只需要索引
        neigh_idx = knn.kneighbors([X_min[idx]], return_distance=False)[0]
        
        # 从近邻索引中删除基准样本自身（第一个元素是自身）
        neigh_idx = np.delete(neigh_idx, 0)
        
        # 如果没有其他近邻（理论上不会发生，因为前面已保证len(X_min)>=2）
        if len(neigh_idx) == 0:
            continue
        
        # 从近邻中随机选择一个样本
        neighbor = X_min[np.random.choice(neigh_idx)]
        
        # 生成0-1之间的随机权重
        alpha = np.random.rand()
        
        # 通过线性插值生成合成样本：基准样本和近邻样本的加权平均
        synthetic.append(alpha * X_min[idx] + (1 - alpha) * neighbor)
    
    # 如果成功生成了合成样本
    if synthetic:
        # 将原始特征与合成特征组合
        X_syn = np.vstack([X, np.array(synthetic)])
        # 将原始标签与合成样本标签（少数类标签）组合
        y_syn = np.hstack([y, [minority_label] * len(synthetic)])
        return X_syn, y_syn
    
    # 如果没有生成合成样本，返回原始数据
    return X, y


# ------------------------------------------------------------------
# 数据采样策略
# 功能：提供多种数据采样策略接口，用于处理类别不平衡问题
# ------------------------------------------------------------------
def apply_sampling(X, y, strategy='smote'):
    """
    应用不同的数据采样策略处理类别不平衡问题

    Args:
        X: 特征数据，二维数组格式 (样本数, 特征数)
        y: 标签数据，一维数组格式 (样本数,)
        strategy: 采样策略，可选值包括：
                  'none': 不进行采样
                  'smote': 使用SMOTE算法过采样少数类
                  'undersample': 随机下采样多数类
                  'combined': 结合SMOTE和ENN的采样方法
                  'knn': 使用自定义的KNN插值法生成少数类样本

    Returns:
        采样后的特征数据和标签数据
    """
    if strategy == 'none':
        # 不进行采样，直接返回原始数据
        return X, y
    elif strategy == 'smote':
        # 使用SMOTE算法过采样少数类
        # 注意：确保近邻数不超过少数类样本数-1
        smote = SMOTE(random_state=42, k_neighbors=min(5, np.sum(y == 1) - 1))
        return smote.fit_resample(X, y)
    elif strategy == 'undersample':
        # 使用随机下采样多数类
        rus = RandomUnderSampler(random_state=42)
        return rus.fit_resample(X, y)
    elif strategy == 'combined':
        # 使用SMOTEENN：先过采样少数类，再用ENN（编辑最近邻）方法清理样本
        sme = SMOTEENN(random_state=42)
        return sme.fit_resample(X, y)
    elif strategy == 'knn':
        # 使用自定义的KNN插值法生成少数类样本
        return augment_minority_knn(X, y, n_samples=1000)
    else:
        # 未知策略时抛出异常
        raise ValueError('Unknown strategy')


def construct_process_pipeline(num_cols: list, cat_cols: list) -> ColumnTransformer:
    # 构建数值型特征处理管道（使用Pipeline串联多个处理步骤）
    numeric_pipe = Pipeline([
        # 缺失值处理：用该特征的中位数填充缺失值（对异常值不敏感，适合数值特征）
        # 这里作为"双重保障"，处理可能在之前预处理中未完全处理的剩余缺失值
        ('imputer', SimpleImputer(strategy='median'))
    ])

    # 构建类别型特征处理管道
    categoric_pipe = Pipeline([
        # 缺失值处理：用该特征的众数（出现频率最高的值）填充缺失值（适合类别特征）
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # 独热编码：将类别特征转换为哑变量（如"性别"→"性别_男"/"性别_女"）
        # handle_unknown='ignore'：测试集中出现训练集未见过的类别时，不报错且该类别对应哑变量均为0
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    # 构建完整的列转换器（将不同处理管道应用到对应的特征列）
    process = ColumnTransformer(
        [
            # 对数值特征应用数值处理管道，指定处理的列名为num_cols
            ('num', numeric_pipe, num_cols),
            # 对类别特征应用类别处理管道，指定处理的列名为cat_cols
            ('cat', categoric_pipe, cat_cols),
        ],
        remainder='drop',  # 对于未在num_cols和cat_cols中的列，直接丢弃（不纳入模型）
        verbose_feature_names_out=False,  # 输出特征名称时不包含管道名称（如'num__imputer'→'imputer'）
    )
    return process
