import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from typing import Optional, Union, Any

# ------------------------------------------------------------------
# 贝叶斯优化
# ------------------------------------------------------------------
lgb_space = {
    'n_estimators': Integer(300, 1500),
    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'num_leaves': Integer(20, 100),
    'max_depth': Integer(3, 10),
    'min_child_samples': Integer(10, 100),
    'reg_alpha': Real(0.1, 1.0),
    'reg_lambda': Real(0.1, 1.0),
}

xgb_space = {
    'n_estimators': Integer(300, 1500),
    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'min_child_weight': Integer(1, 10),
    'gamma': Real(0, 0.5),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'reg_alpha': Real(0.1, 1.0),
    'reg_lambda': Real(0.1, 1.0),
}

cat_space = {
    'iterations': Integer(300, 1500),
    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),
    'depth': Integer(3, 10),
    'l2_leaf_reg': Real(0.1, 10.0),
}

# 1. 逻辑回归 (Logistic Regression)
lr_space = {
    'C': Real(1e-3, 1e2, prior='log-uniform'), # 正则化强度
    'penalty': Categorical(['l2']),            # lbfgs solver 只支持 l2 或 None
    'solver': Categorical(['lbfgs', 'liblinear']) 
}

# 2. 支持向量机 (SVM)
svc_space = {
    'C': Real(1e-3, 1e2, prior='log-uniform'),
    'kernel': Categorical(['linear', 'rbf']),  # 常用核函数
    'gamma': Categorical(['scale', 'auto']),   # RBF核参数
}

# 3. 决策树 (Decision Tree)
dt_space = {
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'criterion': Categorical(['gini', 'entropy'])
}

# 4. 随机森林 (Random Forest)
rf_space = {
    'n_estimators': Integer(100, 500),
    'max_depth': Integer(3, 10),
    'min_samples_split': Integer(2, 20),
    'max_features': Categorical(['sqrt', 'log2']),
    'bootstrap': Categorical([True, False])
}

# 5. 梯度提升树 (GBDT - Sklearn原生)
gbdt_space = {
    'n_estimators': Integer(100, 500),
    'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'subsample': Real(0.6, 1.0),
    'min_samples_split': Integer(2, 20)
}

# 6. 朴素贝叶斯 (Gaussian Naive Bayes)
# 朴素贝叶斯参数很少，主要是方差平滑
nb_space = {
    'var_smoothing': Real(1e-9, 1e-5, prior='log-uniform')
}


class TunedMLP(MLPClassifier):
    def __init__(self, layer_idx=0, activation='relu', alpha=0.0001, 
                 learning_rate_init=0.001, max_iter=200, early_stopping=False, 
                 random_state=None):

        # 必须将参数绑定到 self 上，且名称与 __init__ 参数名完全一致
        self.layer_idx = layer_idx
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state

        # 定义层结构映射
        self.layer_configs = [
            (50,),      # index 0
            (100,),     # index 1
            (50, 50)    # index 2
        ]

        super().__init__(
            hidden_layer_sizes=self.layer_configs[0],
            activation=activation,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=random_state
        )

    def fit(self, X, y):
        # 关键步骤：每次 fit 前，根据当前的 layer_idx 更新 hidden_layer_sizes
        idx = int(self.layer_idx)
        # 边界保护
        if idx < 0 or idx >= len(self.layer_configs):
            idx = 0
        self.hidden_layer_sizes = self.layer_configs[idx]

        return super().fit(X, y)


# 7. 多层感知机 (MLP / Neural Network)
mlp_space = {
    'layer_idx': Categorical([0, 1, 2]),  # 对应上面的 configs
    'activation': Categorical(['tanh', 'relu']),
    'alpha': Real(1e-5, 1e-2, prior='log-uniform'),
    'learning_rate_init': Real(1e-4, 0.01, prior='log-uniform')
}


def bayes_opt_serial(model, space, X, y, n_iter=12):
    opt = BayesSearchCV(
        model, space, n_iter=n_iter, cv=3, scoring='f1',
        random_state=42, n_jobs=1, verbose=0  # 单进程
    )
    opt.fit(X, y)
    return opt.best_estimator_, opt.best_params_


def bayes_opt(model, space, X, y, cv=3, n_iter=12, scoring='f1', random_state=42):
    """通用贝叶斯优化"""
    opt = BayesSearchCV(
        model, space, n_iter=n_iter, cv=cv, scoring=scoring,
        random_state=random_state, n_jobs=-1, verbose=0
    )
    opt.fit(X, y)
    return opt.best_estimator_, opt.best_params_


# ------------------------------------------------------------------
# 最优化F1阈值
# ------------------------------------------------------------------
def f1_best_threshold(y_true, y_prob):
    """返回最大化 F1 的阈值与对应 F1"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1)
    return thresholds[best_idx], f1[best_idx]


def get_best_f1(y_true, y_prob):
    threshold, f1 = f1_best_threshold(y_true, y_prob)
    return f1
