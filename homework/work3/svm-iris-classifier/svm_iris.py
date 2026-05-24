import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.decomposition import PCA


def main():
    # 加载鸢尾花数据集
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    target_names = iris.target_names
    feature_names = iris.feature_names

    print("=" * 55)
    print("SVM 鸢尾花分类器")
    print("=" * 55)
    print(f"\n样本数: {X.shape[0]}")
    print(f"特征数: {X.shape[1]} ({', '.join(feature_names)})")
    print(f"类别: {target_names.tolist()}")
    print(f"\n各类别样本数:")
    for name, count in zip(target_names, np.bincount(y)):
        print(f"  {name}: {count}")

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. 标准化（SVM 对特征尺度敏感）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 网格搜索 + 交叉验证找最优参数

    param_grid = {
        "C": [0.1, 1, 10, 50, 100],
        "gamma": ["scale", "auto", 0.01, 0.1, 0.5],
        "kernel": ["rbf", "linear", "poly"],
    }

    grid_search = GridSearchCV(
        SVC(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    grid_search.fit(X_train_scaled, y_train)

    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"交叉验证最高准确率: {grid_search.best_score_:.4f}")

    # 5. 用最优模型预测
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    # 6. 模型评估
    print("\n" + "-" * 55)
    print("测试集评估结果")
    print("-" * 55)
    print(f"\n测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))

   # 7. 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("混淆矩阵已保存至 confusion_matrix.png")

    # 8. PCA 降维可视化（2D）
    print("\n" + "-" * 55)
    print("PCA 降维可视化")
    print("-" * 55)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # 在 PCA 空间上训练一个用于可视化的 SVM
    X_train_pca, X_test_pca, y_train_p, y_test_p = train_test_split(
        X_pca, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler_pca = StandardScaler()
    X_train_pca_s = scaler_pca.fit_transform(X_train_pca)
    X_test_pca_s = scaler_pca.transform(X_test_pca)

    vis_svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
    vis_svm.fit(X_train_pca_s, y_train_p)

    # 绘制决策边界
    x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
    y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_scaled = scaler_pca.transform(grid_points)
    Z = vis_svm.predict(grid_scaled)
    Z = Z.reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：真实标签分布
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    for i, name in enumerate(target_names):
        axes[0].scatter(
            X_pca[y == i, 0], X_pca[y == i, 1],
            c=colors[i], label=name, edgecolors="k", s=50, alpha=0.8
        )
    axes[0].set_title("True Labels (PCA projection)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 子图2：SVM 决策边界
    axes[1].contourf(xx, yy, Z, alpha=0.3, colors=colors)
    for i, name in enumerate(target_names):
        axes[1].scatter(
            X_pca[y == i, 0], X_pca[y == i, 1],
            c=colors[i], label=name, edgecolors="k", s=50, alpha=0.8
        )
    axes[1].set_title("SVM Decision Boundaries (rbf kernel, PCA)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_visualization.png", dpi=150)
    plt.close()
    print("PCA 可视化已保存至 pca_visualization.png")

    

    print("\n" + "=" * 55)
    print("分类完成！")
    print("=" * 55)


if __name__ == "__main__":
    main()
