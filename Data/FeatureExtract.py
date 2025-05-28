import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 读取数据，注意编码问题
df = pd.read_csv("cleaned_tweets.csv", encoding='ISO-8859-1')

# 准备文本和标签
X_text = df["clean_text"].dropna()
y = df.loc[X_text.index, "airline_sentiment"]

# TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X_text)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("TF-IDF shape:", X.shape)
print("训练集样本数:", X_train.shape[0])
print("测试集样本数:", X_test.shape[0])

# 定义SVM模型
svm = SVC(random_state=42)

# 定义参数网格，尝试不同的C和核函数
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

# 使用网格搜索交叉验证，寻找最优参数
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("最佳参数：", grid_search.best_params_)

# 用最优模型预测测试集
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# 输出准确率和分类报告
print("准确率 (Accuracy):", accuracy_score(y_test, y_pred))
print("\n分类报告:\n", classification_report(y_test, y_pred))

# 类别标签
classes = best_model.classes_

# 混淆矩阵可视化
cm = confusion_matrix(y_test, y_pred, labels=classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Precision, Recall, F1-score可视化
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=classes)

x = np.arange(len(classes))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, precision, width, label='Precision')
plt.bar(x, recall, width, label='Recall')
plt.bar(x + width, f1, width, label='F1-score')

plt.xticks(x, classes)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Classification Metrics by Class')
plt.legend()
plt.show()

# 类别分布可视化
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Class Distribution in Dataset')
plt.xlabel('Sentiment Class')
plt.ylabel('Number of Samples')
plt.show()

# 输出训练测试集的类别分布
print(f"训练集样本数: {X_train.shape[0]}")
print(f"测试集样本数: {X_test.shape[0]}")
print(f"训练集类别分布:\n{y_train.value_counts()}")
print(f"测试集类别分布:\n{y_test.value_counts()}")
