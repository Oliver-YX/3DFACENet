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


# train_file_path = 'your_train_coef_path'
# test_file_path = 'your_test_coef_path'
all_file_path = 'your_all_coef_path'
save_path = 'your_save_path_for_visualization'

if not os.path.exists(save_path):
    os.makedirs(save_path)


# Define a function to classify scores into 'beauty' and 'normal'
def classify_score(score):
    return 'beauty' if score > 4.0 else 'normal'

# # load Excel file
# train_data = pd.read_excel(train_file_path)
# test_data = pd.read_excel(test_file_path)
all_data = pd.read_excel(all_file_path)

# # apply the classification to both datasets
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
    # label mapping
    label_mapping = {}
    # for cluster_label in [0, 1]:
    #     print(cluster_label)
    #     mask = labels == cluster_label
    #     most_common = mode(df[mask]['category']).mode[0]
    #     print(most_common)
    #     label_mapping[cluster_label] = most_common
    
    label_mapping[0] = 'normal'
    label_mapping[1] = 'beauty'
    # map cluster labels to actual categories
    mapped_labels = [label_mapping[label] for label in labels]

    # calculate accuracy
    accuracy = accuracy_score(df['category'], mapped_labels)
    print(f"聚类准确率: {accuracy}")

     # calculate confusion matrix
    cm = confusion_matrix(df['category'], mapped_labels, labels=["beauty", "normal"])
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["beauty", "normal"], yticklabels=["beauty", "normal"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # save image
    plt.savefig(os.path.join(save_path,f"confusion_matrix{name}"))

    plt.show()

    # plot scatter plot and clustering centers in two dimensions
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
    # use all 80 dimensions for clustering
    features = [f'id_{i}' for i in range(1, 81)]
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df[features])
    centers_all = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # label mapping
    label_mapping = {}
    # for cluster_label in [0, 1]:
    #     mask = labels == cluster_label
    #     most_common = mode(df[mask]['category']).mode[0]
    #     label_mapping[cluster_label] = most_common
    label_mapping[0] = 'normal'
    label_mapping[1] = 'beauty'
    
    # map cluster labels to actual categories
    mapped_labels = [label_mapping[label] for label in labels]

    # calculate accuracy
    accuracy = accuracy_score(df['category'], mapped_labels)
    print(f"clustering accuracy: {accuracy}")

    # use PCA to reduce data to 2 dimensions for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df[features])
    plot_data = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    plot_data['category'] = df['category']
    plot_data['cluster'] = labels

    # plot scatter plot and clustering centers in two dimensions
    plt.figure(figsize=(10, 6))
    # sns.scatterplot(x='PC1', y='PC2', hue='cluster', style='category', sizes=(20, 100), data=plot_data, palette='bright')
    sns.scatterplot(x='PC1', y='PC2', hue='category', sizes=(20, 100), data=plot_data, palette='bright')
    centers = pca.transform(centers_all)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='black', marker='X')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(save_path, name))

    
    # calculate confusion matrix
    cm = confusion_matrix(df['category'], mapped_labels, labels=["beauty", "normal"])
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["beauty", "normal"], yticklabels=["beauty", "normal"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # save image
    plt.savefig(os.path.join(save_path,f"confusion_matrix{name}"))

    plt.show()
    
    # classify clustering centers
    # beauty_centers = centers_all[labels == 0]
    # normal_centers = centers_all[labels == 1]
    beauty_centers = [centers_all[i] for i in range(2) if classify_score(centers_all[i][0]) == 'beauty']
    normal_centers = [centers_all[i] for i in range(2) if classify_score(centers_all[i][0]) == 'normal']


    return beauty_centers, normal_centers

def plot_scatter_and_centers_all(df, title, save_path, name):
    # use all 80 dimensions for clustering
    shape_features = [f'id_{i}' for i in range(1, 81)]
    color_features = [f'tex_{i}' for i in range(1,81)]
    features = shape_features + color_features
    kmeans = KMeans(n_clusters=2, random_state=0).fit(df[features])
    centers_all = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    # label mapping
    label_mapping = {}
    # for cluster_label in [0, 1]:
    #     mask = labels == cluster_label
    #     most_common = mode(df[mask]['category']).mode[0]
    #     label_mapping[cluster_label] = most_common
    label_mapping[1] = 'normal'
    label_mapping[0] = 'beauty'
    
    # map cluster labels to actual categories
    mapped_labels = [label_mapping[label] for label in labels]

    # calculate accuracy
    accuracy = accuracy_score(df['category'], mapped_labels)
    print(f"clustering accuracy: {accuracy}")

    # use PCA to reduce data to 2 dimensions for visualization
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df[features])
    plot_data = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    plot_data['category'] = df['category']
    plot_data['cluster'] = labels

    # plot scatter plot and clustering centers in two dimensions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='PC1', y='PC2', hue='cluster', style='category', sizes=(20, 100), data=plot_data, palette='bright')
    # sns.scatterplot(x='PC1', y='PC2', hue='category', sizes=(20, 100), data=plot_data, palette='bright')
    centers = pca.transform(centers_all)
    plt.scatter(centers[:, 0], centers[:, 1], s=100, c='black', marker='X')
    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.savefig(os.path.join(save_path, name))

    
    # calculate confusion matrix
    cm = confusion_matrix(df['category'], mapped_labels, labels=["beauty", "normal"])
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["beauty", "normal"], yticklabels=["beauty", "normal"])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # save image
    plt.savefig(os.path.join(save_path,f"confusion_matrix{name}"))

    plt.show()
    
    # # classify clustering centers
    # beauty_centers = centers_all[labels == 0]
    # normal_centers = centers_all[labels == 1]
    beauty_centers = [centers_all[i] for i in range(2) if classify_score(centers_all[i][0]) == 'beauty']
    normal_centers = [centers_all[i] for i in range(2) if classify_score(centers_all[i][0]) == 'normal']


    return beauty_centers, normal_centers


def plot_beauty_scatter_and_centers_all(df, title, save_path, name, max_clusters):
    all_centers = []
    wcss = []  # used to store the total within-cluster sum of squares for each K value
    silhouette_scores = []  # used to store the silhouette score for each K value
    shape_features = [f'id_{i}' for i in range(1, 81)]
    color_features = [f'tex_{i}' for i in range(1,81)]
    features = shape_features + color_features
    beauty_data = df[df['category'] == 'beauty']

    # iterate different number of clusters
    for n_clusters in range(1, max_clusters + 1):
        # KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(beauty_data[features])
        labels = kmeans.labels_
        all_centers.append(kmeans.cluster_centers_)

        # calculate total within-cluster sum of squares
        wcss.append(kmeans.inertia_)

        # calculate silhouette score
        if n_clusters > 1:
            silhouette_scores.append(silhouette_score(beauty_data[features], labels))

        # PCA dimensionality reduction
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(beauty_data[features])
        
        # prepare scatter plot data
        plot_data = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
        plot_data['cluster'] = labels
        
        # plot scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=plot_data, palette='viridis')
        plt.title(f'{title}', fontsize=22)
        plt.xlabel('PC1', fontsize=20)
        plt.ylabel('PC2', fontsize=20)
        plt.savefig(os.path.join(save_path, f'{name}_{n_clusters}_clusters_scatter.png'))
        plt.close()

    return all_centers


######### cluster in shape and texture dimensions
# AF_centers = plot_scatter_and_centers_all(AF_data, 'Asian Female Scatter Plot with Cluster Centers',save_path,'AF.png')
# AM_centers = plot_scatter_and_centers_all(AM_data, 'Aisan Male Scatter Plot with Cluster Centers',save_path,'AM.png')
# CF_centers = plot_scatter_and_centers_all(CF_data, 'Caucasia Female Scatter Plot with Cluster Centers',save_path,'CF.png')
# CM_centers = plot_scatter_and_centers_all(CM_data, 'Caucasia  Male Scatter Plot with Cluster Centers',save_path,'CM.png')

# F_centers = plot_scatter_and_centers_all(F_data, 'Female Scatter Plot with Cluster Centers',save_path,'F.png')
# M_centers = plot_scatter_and_centers_all(M_data, 'Male Scatter Plot with Cluster Centers',save_path,'M.png')

# # ########## cluster in shape dimensions 
# AF_centers = plot_scatter_and_centers_shape(AF_data, 'Asian Female Scatter Plot with Cluster Centers',save_path,'AF.png')
# AM_centers = plot_scatter_and_centers_shape(AM_data, 'Aisan Male Scatter Plot with Cluster Centers',save_path,'AM.png')
# CF_centers = plot_scatter_and_centers_shape(CF_data, 'Caucasia Female Scatter Plot with Cluster Centers',save_path,'CF.png')
# CM_centers = plot_scatter_and_centers_shape(CM_data, 'Caucasia  Male Scatter Plot with Cluster Centers',save_path,'CM.png')

# F_centers = plot_scatter_and_centers_shape(F_data, 'Female Scatter Plot with Cluster Centers',save_path,'F.png')
# M_centers = plot_scatter_and_centers_shape(M_data, 'Male Scatter Plot with Cluster Centers',save_path,'M.png')

# train_info_path = f"{save_path}/center.txt"
# with open(train_info_path, 'w') as file:
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