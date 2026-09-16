import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay,
)
from typing import Optional, Union, Any
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap


def plot_label_distribution(
    df: pd.DataFrame,
    label_desc: str,
    pie_title: Optional[str] = None,
    bar_title: Optional[str] = None,
) -> None:
    # 创建画布和子图，调整子图间距
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))  # 创建画布和子图
    plt.subplots_adjust(wspace=0.3)  # 调整子图间距

    # 定义配色方案
    colors = ['#4CAF50', '#FF5252']

    # 1. 饼图优化 - 展示样本占比
    pie_title = pie_title or f"{label_desc} distribution ratio".capitalize()
    df[label_desc].value_counts().plot.pie(
        explode=[0.1] * df[label_desc].nunique(),
        autopct='%1.1f%%',
        ax=ax[0],
        shadow=True,
        colors=colors,
        startangle=90,
        textprops={'fontsize': 14, 'color': '#333333'}
    )
    ax[0].set_title(pie_title, fontsize=16, pad=20, fontweight='bold')  # 修改标题
    ax[0].set_ylabel('')  # 去除y轴标签
    ax[0].axis('equal')  # 保证饼图是正圆形

    # 2. 柱状图优化 - 展示具体数量
    bar_title = bar_title or f"Distribution of {label_desc}".capitalize()
    sns.countplot(
        x=label_desc,
        data=df,
        ax=ax[1],
        palette=colors,
        width=0.6
    )
    ax[1].set_title(bar_title, fontsize=16, pad=20, fontweight='bold') # 修改标题
    ax[1].set_xlabel('Label Category', fontsize=14, labelpad=10)   # 修改X轴
    ax[1].set_ylabel('Quantity', fontsize=14, labelpad=10)  # 修改Y轴

    # 为柱状图添加数值标签
    for p in ax[1].patches:
        height = p.get_height()
        ax[1].text(
            p.get_x() + p.get_width() / 2.,  # x坐标
            height + 5,  # y坐标（在柱顶上方）
            f'{int(height)}',  # 显示的数值
            ha='center', va='bottom', fontsize=12, color='#333333'
        )

    # 美化坐标轴刻度
    ax[1].tick_params(axis='both', which='major', labelsize=12)

    # 去除顶部和右侧边框
    for spine in ax[1].spines.values():
        if spine.spine_type in ['top', 'right']:
            spine.set_visible(False)

    # plt.show()


def plot_discrete_variable_label_relationship(
    df: pd.DataFrame,
    var: str,
    label: str,
    label_desc: str,
) -> None:
    """
    离散变量与标签的关系

    Args:
        df (pd.DataFrame): 数据
        var (str): 变量名
        label (str): 标签名
        label_desc (str): 标签描述名

    Returns:
        None
    """
    n_label = df[label].nunique()
    label0, label1 = dict(df[[label, label_desc]].drop_duplicates().sort_values(by=label).values).values()

    fig, axes = plt.subplots(1, n_label, figsize=(8 * n_label, 7))
    axes = axes.flatten()

    # 定义颜色
    mean_color = '#2C7FB8'
    count_colors = ['#28A745', '#DC3545']

    # --- 均值图 ---
    feature_mean = df[[var, label]].groupby([var]).mean().sort_index()
    feature_mean.plot.bar(
        ax=axes[0],
        color=mean_color,
        width=0.6,
        edgecolor='white',
        linewidth=1.2,
        alpha=0.9,
        legend=False,
    )

    axes[0].set_title(f'{var} vs {label1} Rates', fontsize=14, pad=15, fontweight='bold')
    axes[0].set_ylabel(f'{label1} Rates', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45, labelsize=9)

    # 均值标签
    for p in axes[0].patches:
        axes[0].text(
            p.get_x() + p.get_width()/2, p.get_height() + 0.005, f'{p.get_height():.2f}', 
            ha='center', va='bottom', fontsize=9, color='#555555')

    # --- 右侧：计数图 ---
    sns.countplot(
        x=var, hue=label_desc, data=df, ax=axes[1], palette=count_colors, 
        width=0.7, edgecolor='white', linewidth=1.2, order=feature_mean.index)

    axes[1].set_title(f'{var} Distribution', fontsize=14, pad=15, fontweight='bold')
    axes[1].set_ylabel('Sample Size', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45, labelsize=9)
    axes[1].legend(title='Label', labels=[label0, label1], fontsize=9, title_fontsize=10, loc=(0.95, 0.95), frameon=False)
    # axes[1].legend(title='Label', labels=[label0, label1], fontsize=9, title_fontsize=10, loc='upper right', frameon=False)

    # 数量标签
    for p in axes[1].patches:
        h = p.get_height()
        if h > 0:
            axes[1].text(p.get_x() + p.get_width()/2, h + 5, f'{int(h)}', ha='center', va='bottom', fontsize=9)

    # 美化共有部分：去除边框
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel('')  # 隐藏多余的下标签，由标题体现

    plt.tight_layout()
    # plt.show()


def plot_continuous_variable_label_relationship(
    df: pd.DataFrame,
    var: str,
    label: str,
    label_desc: str,
    bins: Union[list[Union[int, float]], int] = 10,
    names: Optional[list[str]] = None,
):
    """
    连续变量与标签的关系

    Args:
        df (pd.DataFrame): 数据
        var (str): 变量名
        label (str): 标签名
        label_desc (str): 标签描述名
        bins (Union[list[Union[int, float]], int], optional): 分箱数或分箱列表. Defaults to 10.
        names (Optional[list[str]], optional): 分箱标签. Defaults to None.

    Returns:
        None
    """
    if names is not None:
        if isinstance(bins, int):
            assert len(names) == bins , 'names must be None or have length bins - 1'
        else:
            assert len(names) == len(bins) - 1, 'names must be None or have length bins - 1'

    # 创建临时变量列，将连续变量分箱为离散变量
    temp = df[var].copy()
    df[var] = pd.cut(df[var], bins=bins, labels=names, right=False)

    plot_discrete_variable_label_relationship(
        df=df,
        var=var,
        label=label,
        label_desc=label_desc,
    )

    # 恢复原始变量列
    df[var] = temp


def plot_variables_distribution(
    df: pd.DataFrame,
    variables: list[str],
) -> None:
    """
    变量分布图

    Args:
        df (pd.DataFrame): 数据
        variables (list[str]): 变量列表

    Returns:
        None
    """

    n_var = len(variables)

    # 自动计算每行每列的子图数量
    n_cols = int(np.floor(n_var**.5)) + 1
    n_rows = n_var // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*4))
    axes = axes.flatten()

    # 循环绘制分布图
    for ax, feature in zip(axes, variables):
        # 绘制直方图和核密度曲线
        sns.histplot(
            data=df,
            x=feature, 
            bins=30,  # 稍微增加bins数量，因为传感器数据比较连续
            kde=True,
            ax=ax,
            color='#3498db',  # 直方图颜色 (蓝色)
            edgecolor='white',
            linewidth=0.5,
            line_kws={  # 核密度曲线的样式参数
                'color': '#e74c3c',  # 曲线颜色 (红色)
                'linewidth': 2
            }
        )

        # 设置标题和标签 (自动处理单位换行)
        # 将括号内的单位换行显示，避免标题过长
        clean_title = feature.replace(' (', '\n(')
        ax.set_title(clean_title, pad=10, fontweight='bold', fontsize=11)
        ax.set_xlabel('') # x轴标签已在标题中体现，这里留空或简化
        ax.set_ylabel('Frequency', labelpad=5)

        # 美化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', rotation=30)

        # 添加网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)

    # 调整布局
    plt.tight_layout()
    # plt.show()


def plot_heatmap(df: pd.DataFrame) -> None:
    num_df = df.select_dtypes(include=[np.number]).copy()

    # 如果标签列是数值型，也会包含进来；为了与论文一致，你可以保留或移除
    corr = num_df.corr(method='pearson')

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, cmap='coolwarm', center=0, square=False,
        linewidths=0.3, cbar_kws={'shrink': 0.8},
    )
    plt.title('Correlation Heatmap (Numeric Features)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # plt.show()


def plot_roc_pr_curve(true, plot_data):
    """
    绘制 ROC/PR 曲线

    Args:
        true: 真实标签
        plot_data: 数据列表，每个元素为 (y_prob, label, kwargs)
            例如：
            plot_data = [
                (y_prob_lr,   'LR',   {'lw': 1, 'alpha': 0.6, 'linestyle': '--'}),
                (y_prob_lgb,  'LightGBM', {'lw': 2, 'alpha': 0.8}),
            ]
        title: 图表标题
    """
    # 创建画布
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    print("正在绘制 ROC/PR 曲线...")

    # --- ROC 曲线 ---
    ax[0].plot([0, 1], [0, 1], linestyle=':', lw=2, color='gray', label='Random')  # 对角线
    for prob, label, kwargs in plot_data:
        RocCurveDisplay.from_predictions(
            true, prob,
            name=label,
            ax=ax[0],
            plot_chance_level=False,  # 手动画了对角线，这里关掉
            **kwargs
        )

    ax[0].set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc='lower right', fontsize=9)

    # --- PR 曲线 ---
    # 计算基准线 (Positive Rate)
    baseline = true.sum() / len(true)
    ax[1].plot([0, 1], [baseline, baseline], linestyle=':', lw=2, color='gray', label=f'Baseline ({baseline:.2f})')

    for prob, label, kwargs in plot_data:
        PrecisionRecallDisplay.from_predictions(
            true, prob,
            name=label,
            ax=ax[1],
            plot_chance_level=False,
            **kwargs
        )

    ax[1].set_title('Precision-Recall Curve Comparison', fontsize=14, fontweight='bold')
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    # plt.show()


def plot_best_model_analysis(
    true: np.ndarray,
    best_model_name: str,
    best_model_oof: np.ndarray,
    global_best_f1: float,
    global_best_thr: float,
    display_labels: list[str] = ['Class 0', 'Class 1'],
) -> None:
    # ======================================================
    # 绘制最佳单模型的 阈值-F1 曲线
    # ======================================================
    # 重新计算一遍胜出模型的数据用于绘图
    precision, recall, thresholds = precision_recall_curve(true, best_model_oof)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, f1s[:-1], label='F1 Score', color='dodgerblue', linewidth=2)
    plt.axvline(
        global_best_thr, color='red', linestyle='--', alpha=0.8,
        label=f'Best Threshold = {global_best_thr:.3f}'
    )
    plt.axhline(
        global_best_f1, color='orange', linestyle=':', alpha=0.8,
        label=f'Max F1 = {global_best_f1:.4f}'
    )
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title(f'Threshold vs F1 Analysis ({best_model_name})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower center')
    plt.grid(True, alpha=0.3)

    # ======================================================
    # 绘制最佳单模型的 混淆矩阵
    # ======================================================
    # 使用最佳阈值将概率转为 0/1
    pred_best_binary = (best_model_oof >= global_best_thr).astype(int)

    ax = plt.subplot(1, 2, 2)
    cm = confusion_matrix(true, pred_best_binary)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(cmap='Blues', ax=ax, values_format='d', colorbar=False)
    ax.set_title(
        f'Confusion Matrix - {best_model_name}\n(Thr={global_best_thr:.3f}, F1={global_best_f1:.3f})',
        fontsize=13, fontweight='bold'
    )
    # plt.show()
    # plt.close()


def plot_feature_importance(model, feature_names, top_n=30):
    if isinstance(model, lgb.LGBMClassifier):
        model_name = 'LightGBM'
    elif isinstance(model, xgb.XGBClassifier):
        model_name = 'XGBoost'
    elif isinstance(model, cb.CatBoostClassifier):
        model_name = 'CatBoost'
    else:
        raise ValueError(f'Unsupported model type: {type(model)}')

    # 获取特征重要性
    importance = model.feature_importances_

    # 计算Top30（按gain降序）
    imp_df = pd.Series(importance, index=feature_names).sort_values(ascending=False).head(top_n)

    # 仅用于绘图显示：去掉特征名前缀 "num_"
    plot_df = imp_df.reset_index()
    plot_df.columns = ['feature', 'gain']
    plot_df['feature'] = (
        plot_df['feature']
        .str.replace(r'^num__', '', regex=True)
        .str.replace(r'^num_', '', regex=True)
    )

    plt.figure(figsize=(6, 8))
    sns.barplot(data=plot_df, x='gain', y='feature', order=plot_df['feature'])
    plt.title(f'Top{top_n} {model_name} Gain')
    plt.xlabel('Gain')
    # plt.show()
    # plt.close()
