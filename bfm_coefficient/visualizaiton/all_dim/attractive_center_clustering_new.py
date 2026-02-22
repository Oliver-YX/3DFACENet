from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
from sklearn.metrics import silhouette_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os



# train_file_path = "your_location/train_coef.xlsx"
# test_file_path = "your_location/test_coef.xlsx"
YOUR_COEF_ROOT = "your_location"
all_file_path = os.path.join(YOUR_COEF_ROOT, "all_coef.xlsx")
YOUR_VIZ_SAVE_PATH = "your_location/visualizaiton/all_dim/beauty/4.0"
save_path = YOUR_VIZ_SAVE_PATH

if not os.path.exists(save_path):
    os.makedirs(save_path)


# Define a function to classify scores into 'beauty' and 'normal'
def classify_score(score):
    return 'beauty' if score > 4.0 else 'normal'

# Read coefficient Excel
# train_data = pd.read_excel(train_file_path)
# test_data = pd.read_excel(test_file_path)
all_data = pd.read_excel(all_file_path)
# Apply the classification to both datasets
# train_data['category'] = train_data['score'].apply(classify_score)
# test_data['category'] = test_data['score'].apply(classify_score)
all_data['category'] = all_data['score'].apply(classify_score)
AF_data = all_data[all_data.iloc[:, 0].str.contains('AF')]
AM_data = all_data[all_data.iloc[:, 0].str.contains('AM')]
CF_data = all_data[all_data.iloc[:, 0].str.contains('CF')]
CM_data = all_data[all_data.iloc[:, 0].str.contains('CM')]
F_data = all_data[all_data.iloc[:, 0].str.contains('F')]
M_data = all_data[all_data.iloc[:, 0].str.contains('M')]

# Function to plot scatter plot and clustering centers in two dimensions
def plot_scatter_and_centers(df, title, save_path, name):
    # Selecting only the relevant columns
    plot_data = df[['id_41', 'id_6', 'category']]

    # KMeans clustering with 2 centers (beauty and normal)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(plot_data[['id_41', 'id_6']])
    labels = kmeans.labels_
    print(labels)
    # 标签映射
    label_mapping = {}
    # for cluster_label in [0, 1]:
    #     print(cluster_label)
    #     mask = labels == cluster_label
    #     most_common = mode(df[mask]['category']).mode[0]
    #     print(most_common)
    #     label_mapping[cluster_label] = most_common
    
    label_mapping[0] = 'normal'
    label_mapping[1] = 'beauty'
    # Map cluster labels to category
    mapped_labels = [label_mapping[label] for label in labels]

    # Compute accuracy
    accuracy = accuracy_score(df['category'], mapped_labels)
    print(f"Clustering accuracy: {accuracy}")

    # Confusion matrix
    cm = confusion_matrix(df['category'], mapped_labels, labels=["beauty", "normal"])
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["beauty", "normal"], yticklabels=["beauty", "normal"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save figure
    plt.savefig(os.path.join(save_path, f"confusion_matrix{name}"))

    plt.show()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='id_41', y='id_6', hue='category', sizes = (20, 100), data=plot_data, palette='bright')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='black', marker='X')
    plt.title(title)
    plt.xlabel('id_41')
    plt.ylabel('id_6')
    plt.savefig(os.path.join(save_path, name))
    
    beauty_centers = [centers[i] for i in range(2) if classify_score(centers[i][0]) == 'beauty']
    normal_centers = [centers[i] for i in range(2) if classify_score(centers[i][0]) == 'normal']

    return beauty_centers, normal_centers

def plot_scatter_and_centers_shape(df, title, save_path, name):
    # 使用所有80维特征进行聚类
    features = [f'id_{i}' for i in range(1, 81)]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df[features])
    centers_all = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # 标签映射
    label_mapping = {}
    # for cluster_label in [0, 1]:
    #     mask = labels == cluster_label
    #     most_common = mode(df[mask]['category']).mode[0]
    #     label_mapping[cluster_label] = most_common
    label_mapping[0] = 'normal'
    label_mapping[1] = 'beauty'
    
    # 将聚类标签映射到实际类别
    mapped_labels = [label_mapping[label] for label in labels]

    # 计算准确率
    accuracy = accuracy_score(df['category'], mapped_labels)
    print(f"聚类准确率: {accuracy}")

    # 使用PCA将数据降至2维以进行可视化
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df[features])
    plot_data = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    plot_data['category'] = df['category']
    plot_data['cluster'] = labels

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    # sns.scatterplot(x='PC1', y='PC2', hue='cluster', style='category', sizes=(20, 100), data=plot_data, palette='bright')
    sns.scatterplot(x='PC1', y='PC2', hue='category', sizes=(20, 100), data=plot_data, palette='bright')
    centers = pca.transform(centers_all)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='black', marker='X')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(save_path, name))

    
    # 计算混淆矩阵
    cm = confusion_matrix(df['category'], mapped_labels, labels=["beauty", "normal"])
    # 绘制混淆矩阵图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["beauty", "normal"], yticklabels=["beauty", "normal"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save figure
    plt.savefig(os.path.join(save_path, f"confusion_matrix{name}"))

    plt.show()
    
    # 分类聚类中心
    # beauty_centers = centers_all[labels == 0]
    # normal_centers = centers_all[labels == 1]
    beauty_centers = [centers_all[i] for i in range(2) if classify_score(centers_all[i][0]) == 'beauty']
    normal_centers = [centers_all[i] for i in range(2) if classify_score(centers_all[i][0]) == 'normal']


    return beauty_centers, normal_centers

def plot_beauty_scatter_and_centers_shape(df, title, save_path, name, max_clusters):
    all_centers = []
    wcss = []  # 用于存储每个K值的总内部平方和
    silhouette_scores = []  # 用于存储每个K值的轮廓系数
    features = [f'id_{i}' for i in range(1, 81)]
    beauty_data = df[df['category'] == 'beauty']

    # 迭代不同数量的聚类
    for n_clusters in range(1, max_clusters + 1):
        # KMeans聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(beauty_data[features])
        labels = kmeans.labels_
        all_centers.append(kmeans.cluster_centers_)

        # 计算总内部平方和
        wcss.append(kmeans.inertia_)

        # 计算轮廓系数
        if n_clusters > 1:
            silhouette_scores.append(silhouette_score(beauty_data[features], labels))

        # PCA降维
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(beauty_data[features])

        # 准备散点图数据
        plot_data = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
        plot_data['cluster'] = labels

        # 绘制散点图
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=plot_data, palette='viridis')
        plt.title(f'{title} - {n_clusters} Clusters')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.savefig(os.path.join(save_path, f'{name}_{n_clusters}_clusters_scatter.png'))
        plt.close()

    # 绘制肘部法则图
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig(os.path.join(save_path, f'{name}_elbow_method.png'))
    plt.close()

    # 绘制轮廓系数图
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')  # 从2开始，因为轮廓系数至少需要两个聚类
    plt.title('Silhouette Coefficient')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join(save_path, f'{name}_silhouette_coefficient.png'))
    plt.close()

    return all_centers

def plot_scatter_and_centers_all(df, title, save_path, name):
    # 使用所有80维特征进行聚类
    shape_features = [f'id_{i}' for i in range(1, 81)]
    color_features = [f'tex_{i}' for i in range(1,81)]
    features = shape_features + color_features
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df[features])
    centers_all = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # 标签映射
    label_mapping = {}
    # for cluster_label in [0, 1]:
    #     mask = labels == cluster_label
    #     most_common = mode(df[mask]['category']).mode[0]
    #     label_mapping[cluster_label] = most_common
    label_mapping[1] = 'normal'
    label_mapping[0] = 'beauty'
    
    # 将聚类标签映射到实际类别
    mapped_labels = [label_mapping[label] for label in labels]

    # 计算准确率
    accuracy = accuracy_score(df['category'], mapped_labels)
    print(f"聚类准确率: {accuracy}")

    # 使用PCA将数据降至2维以进行可视化
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df[features])
    plot_data = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    plot_data['category'] = df['category']
    plot_data['cluster'] = labels

    # 绘制散点图
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', style='category', sizes=(20, 100), data=plot_data, palette='bright')
    # sns.scatterplot(x='PC1', y='PC2', hue='category', sizes=(20, 100), data=plot_data, palette='bright')
    centers = pca.transform(centers_all)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='black', marker='X')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(save_path, name))

    
    # 计算混淆矩阵
    cm = confusion_matrix(df['category'], mapped_labels, labels=["beauty", "normal"])
    # 绘制混淆矩阵图
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["beauty", "normal"], yticklabels=["beauty", "normal"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save figure
    plt.savefig(os.path.join(save_path, f"confusion_matrix{name}"))

    plt.show()
    
    # Cluster centers
    # beauty_centers = centers_all[labels == 0]
    # normal_centers = centers_all[labels == 1]
    beauty_centers = [centers_all[i] for i in range(2) if classify_score(centers_all[i][0]) == 'beauty']
    normal_centers = [centers_all[i] for i in range(2) if classify_score(centers_all[i][0]) == 'normal']


    return beauty_centers, normal_centers

def plot_beauty_scatter_and_centers_all(df, title, save_path, name, max_clusters):
    all_centers = []
    wcss = [] 
    silhouette_scores = [] 
    shape_features = [f'id_{i}' for i in range(1, 81)]
    color_features = [f'tex_{i}' for i in range(1,81)]
    features = shape_features + color_features
    beauty_data = df[df['category'] == 'beauty']

    # Iterate different number of clusters
    for n_clusters in range(1, max_clusters + 1):
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(beauty_data[features])
        labels = kmeans.labels_
        all_centers.append(kmeans.cluster_centers_)

        # Calculate total within-cluster sum of squares
        wcss.append(kmeans.inertia_)

        # Calculate silhouette score
        if n_clusters > 1:
            silhouette_scores.append(silhouette_score(beauty_data[features], labels))

        # PCA dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(beauty_data[features])
        
        # Prepare scatter plot data
        plot_data = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
        plot_data['cluster'] = labels
        
        # Plot scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=plot_data, palette='viridis')
        # plt.title(f'{title} - {n_clusters} Clusters')
        plt.title(f'{title}', fontsize=22)
        plt.xlabel('PC1', fontsize=20)
        plt.ylabel('PC2', fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig(os.path.join(save_path, f'{name}_{n_clusters}_clusters_scatter.png'))
        plt.close()

        # # TSNE dimensionality reduction
        # tsne = TSNE(n_components=2, random_state=0)
        # reduced_data = tsne.fit_transform(beauty_data[features])
        # # Prepare scatter plot data
        # plot_data = pd.DataFrame(reduced_data, columns=['TSNE1', 'TSNE2'])
        # # plot_data['category'] = df['category'].iloc[:len(reduced_data)]  # 使用与 reduced_data 相同长度的 category 列
        # plot_data['cluster'] = labels 
    
        # # Plot scatter plot
        # plt.figure(figsize=(10, 6))
        # sns.scatterplot(x='TSNE1', y='TSNE2', hue='cluster', data=plot_data, palette='bright')
        
        # # Map cluster centers to t-SNE space
        # # centers_tsne = tsne.fit_transform(centers_all)
        # # plt.scatter(centers_tsne[:, 0], centers_tsne[:, 1], s=100, c='black', marker='X')

        # plt.title(title, fontsize=18)
        # plt.xlabel('TSNE1', fontsize=14)
        # plt.ylabel('TSNE2', fontsize=14)
        # plt.legend(fontsize=12)
        # plt.savefig(os.path.join(save_path, f'{name}_{n_clusters}_clusters_scatter.png'))
        # plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.savefig(os.path.join(save_path, f'{name}_elbow_method.png'))
    plt.close()

    # Plot silhouette coefficient
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')  
    plt.title('Silhouette Coefficient')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join(save_path, f'{name}_silhouette_coefficient.png'))
    plt.close()

    return all_centers

######### cluster in two dimensions
# Plotting for train data
# train_centers = plot_scatter_and_centers(train_data, 'Train Data Scatter Plot with Cluster Centers',save_path,'train.png')

# # Plotting for test data
# test_centers = plot_scatter_and_centers(test_data, 'Test Data Scatter Plot with Cluster Centers',save_path,'test.png')

# # Plotting for test data
# AF_centers = plot_scatter_and_centers(AF_data, 'Asian Female Scatter Plot with Cluster Centers',save_path,'AF.png')

# # Plotting for test data
# AM_centers = plot_scatter_and_centers(AM_data, 'Aisan Male Scatter Plot with Cluster Centers',save_path,'AM.png')

# # Plotting for test data
# CF_centers = plot_scatter_and_centers(CF_data, 'Caucasia Female Scatter Plot with Cluster Centers',save_path,'CF.png')

# # Plotting for test data
# CM_centers = plot_scatter_and_centers(CM_data, 'Caucasia  Male Scatter Plot with Cluster Centers',save_path,'CM.png')

# # Plotting for test data
# F_centers = plot_scatter_and_centers(F_data, 'Female Scatter Plot with Cluster Centers',save_path,'F.png')

# # Plotting for test data
# M_centers = plot_scatter_and_centers(M_data, 'Male Scatter Plot with Cluster Centers',save_path,'M.png')


# # ########## cluster in all dimensions 
# # # Plotting for train data
# train_centers = plot_scatter_and_centers_all(train_data, 'Train Data Scatter Plot with Cluster Centers',save_path,'train.png')

# # Plotting for test data
# test_centers = plot_scatter_and_centers_all(test_data, 'Test Data Scatter Plot with Cluster Centers',save_path,'test.png')

# # Plotting for test data
# AF_centers = plot_scatter_and_centers_all(AF_data, 'Asian Female Scatter Plot with Cluster Centers',save_path,'AF.png')

# # Plotting for test data
# AM_centers = plot_scatter_and_centers_all(AM_data, 'Aisan Male Scatter Plot with Cluster Centers',save_path,'AM.png')

# # Plotting for test data
# CF_centers = plot_scatter_and_centers_all(CF_data, 'Caucasia Female Scatter Plot with Cluster Centers',save_path,'CF.png')

# # Plotting for test data
# CM_centers = plot_scatter_and_centers_all(CM_data, 'Caucasia  Male Scatter Plot with Cluster Centers',save_path,'CM.png')

# Plotting for test data
# F_centers = plot_scatter_and_centers_all(F_data, 'Female Scatter Plot with Cluster Centers',save_path,'F.png')

# Plotting for test data
# M_centers = plot_scatter_and_centers_all(M_data, 'Male Scatter Plot with Cluster Centers',save_path,'M.png')

# # ########## cluster in shape dimensions 
# # # Plotting for train data
# train_centers = plot_scatter_and_centers_shape(train_data, 'Train Data Scatter Plot with Cluster Centers',save_path,'train.png')

# # Plotting for test data
# test_centers = plot_scatter_and_centers_shape(test_data, 'Test Data Scatter Plot with Cluster Centers',save_path,'test.png')

# # Plotting for test data
# AF_centers = plot_scatter_and_centers_shape(AF_data, 'Asian Female Scatter Plot with Cluster Centers',save_path,'AF.png')

# # Plotting for test data
# AM_centers = plot_scatter_and_centers_shape(AM_data, 'Aisan Male Scatter Plot with Cluster Centers',save_path,'AM.png')

# # Plotting for test data
# CF_centers = plot_scatter_and_centers_shape(CF_data, 'Caucasia Female Scatter Plot with Cluster Centers',save_path,'CF.png')

# # Plotting for test data
# CM_centers = plot_scatter_and_centers_shape(CM_data, 'Caucasia  Male Scatter Plot with Cluster Centers',save_path,'CM.png')

# # Plotting for test data
# F_centers = plot_scatter_and_centers_shape(F_data, 'Female Scatter Plot with Cluster Centers',save_path,'F.png')

# # Plotting for test data
# M_centers = plot_scatter_and_centers_shape(M_data, 'Male Scatter Plot with Cluster Centers',save_path,'M.png')


# train_info_path = f"{save_path}/center.txt"
# with open(train_info_path, 'w') as file:
#     file.write(f"train_b_centers: {train_centers[0]}\n")
#     file.write(f"train_n_centers: {train_centers[1]}\n")
#     file.write(f"test_b_centers: {test_centers[0]}\n")
#     file.write(f"test_n_centers: {test_centers[1]}\n")
#     file.write(f"AF_b_centers: {AF_centers[0]}\n")
#     file.write(f"AF_n_centers: {AF_centers[1]}\n")
#     file.write(f"AM_b_centers: {AM_centers[0]}\n")
#     file.write(f"AM_n_centers: {AM_centers[1]}\n")
#     file.write(f"CF_b_centers: {CF_centers[0]}\n")
#     file.write(f"CF_n_centers: {CF_centers[1]}\n")
#     file.write(f"CM_b_centers: {CM_centers[0]}\n")
#     file.write(f"CM_n_centers: {CM_centers[1]}\n")
#     file.write(f"F_b_centers: {F_centers[0]}\n")
#     file.write(f"F_n_centers: {F_centers[1]}\n")
#     file.write(f"M_b_centers: {M_centers[0]}\n")
#     file.write(f"M_n_centers: {M_centers[1]}\n")



# # ########## cluster in shape dimensions on beauty people
# # # Plotting for all data

# F_data = all_data[all_data.iloc[:, 0].str.contains('F')]
# M_data = all_data[all_data.iloc[:, 0].str.contains('M')]
bestFcenters = plot_beauty_scatter_and_centers_all(F_data, 'Attractive Female Data Clustering',save_path,'female_beauty',max_clusters=5)
bestMcenters = plot_beauty_scatter_and_centers_all(M_data, 'Attractive Male Data Clustering',save_path,'male_beauty',max_clusters=5)

train_info_path = f"{save_path}/center.txt"
with open(train_info_path, 'w') as file:
    file.write(f"beauty_F_centers: {bestFcenters}\n")
    file.write(f"beauty_M_centers: {bestMcenters}\n")