import numpy as np
import pandas as pd
import krippendorff
import os
from tqdm import tqdm
import pickle
import sklearn
from sklearn.decomposition import KernelPCA
from sklearn import metrics
import matplotlib.pyplot as plt

from config import config, clustering_config
from utils.hierarchical_clustering import agglomerative_clustering, plot_dendrogram
from utils.split_data import split_datasets
from utils.data_clustering import generate_df

import warnings
from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)


#read data
df = pd.read_csv(f"{config['corpus_file']}")


#label matrix
df_annotator_representation = df[['id_text', 'id_annotator', 'label']]
df_annotator_representation = df_annotator_representation.replace(['iro', 'not'], [1.0, 0.0])
df_label_matrix = df_annotator_representation.pivot_table(index='id_annotator', columns='id_text', values='label')
label_matrix = df_label_matrix.to_numpy(na_value=np.nan, dtype='float')

list_annotators = df_label_matrix.index.tolist()
list_annotators = df_label_matrix.index.tolist()
with open(os.path.join(f"{clustering_config['files_dir']}", "list_annotators.pickle"), "wb") as handle:
    pickle.dump(list_annotators, handle, protocol=pickle.HIGHEST_PROTOCOL)

# matrix one-hot encoded
n_labels = clustering_config["n_labels"]
n_instances = len(df_label_matrix.columns)
matrix_one_hot = np.zeros((len(list_annotators), n_labels*n_instances))

for i in tqdm(range(0, len(label_matrix))):
    row = label_matrix[i]
    one_hot_list = []
    for el in row:
        #print (el)
        if el == 0.0:
            one_hot_list.append(1)
            one_hot_list.append(0)
        elif el == 1.0:
            one_hot_list.append(0)
            one_hot_list.append(1)
        else:
            one_hot_list.append(0)
            one_hot_list.append(0)
    matrix_one_hot[i,:] = np.array(one_hot_list)


# distance matrix krippendorff
similarity_matrix_krippendorff = np.zeros((len(label_matrix), len(label_matrix)))

for i in tqdm(range(0, len(label_matrix))):
    for j in range(0, len(label_matrix)):
        with warnings.catch_warnings(record=True) as w:
            try:
                similarity_matrix_krippendorff[i,j] = krippendorff.alpha(reliability_data=[label_matrix[i, :], label_matrix[j, :]], level_of_measurement='nominal')
                if len(w) > 0:
                    similarity_matrix_krippendorff[i, j] = np.nan
                    f = open(f"{clustering_config['files_dir']}/RuntimeWarning.txt", "a")
                    check_matrix = np.delete([label_matrix[i,:], label_matrix[j,:]], np.where(np.isnan([label_matrix[i,:], label_matrix[j,:]])), axis=1)
                    f.write(str(check_matrix) +"\n\n")
                    f.close()
            except ValueError as Argument:
                similarity_matrix_krippendorff[i,j] = 0
                f2 = open(f"{clustering_config['files_dir']}/no_common_items.txt", "a")
                f2.write(str(Argument)+"\n")
                f2.close()

similarity_matrix_krippendorff = np.nan_to_num(similarity_matrix_krippendorff, copy=True, nan=1, posinf=None, neginf=None)
distance_matrix_krippendorff = np.subtract(1, similarity_matrix_krippendorff)
np.save(os.path.join(f"{clustering_config['files_dir']}", "distance_matrix_krippendorff"), distance_matrix_krippendorff)

# distance matrix kpca
n_components = clustering_config["n_components"]
kpca = KernelPCA(n_components=n_components, kernel='cosine')
kpca_annotator_representation = kpca.fit_transform(matrix_one_hot)
distance_matrix_kpca = sklearn.metrics.pairwise_distances(kpca_annotator_representation, metric='euclidean')
np.save(os.path.join(f"{clustering_config['files_dir']}", "distance_matrix_kpca"), distance_matrix_kpca)

#hierarchical clustering
model_agglomerative_krippendorff = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["Krippendorff_distance_threshold"], distance_matrix_krippendorff)
labels_krippendorff = model_agglomerative_krippendorff.labels_
print ("KRIPPENDORFF: Dendrogram sklearn estimated number of clusters: ", model_agglomerative_krippendorff.n_clusters_)

model_agglomerative_kpca = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["KPCA_distance_threshold"], distance_matrix_kpca)
labels_kpca = model_agglomerative_kpca.labels_
print ("KPCA: Dendrogram sklearn estimated number of clusters: ", model_agglomerative_kpca.n_clusters_)


#dendrograms
plt.figure(num=1, figsize=(20,10))
plot_dendrogram(model_agglomerative_krippendorff)
plt.title("Dendrogram Krippendorff's alpha")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(('cluster_3', 'cluster_1', 'cluster_0', 'cluster_4', 'cluster_2'))
plt.xlabel("Unique annotator ID", fontsize=12, labelpad=15)
plt.ylabel("Node level", fontsize=12, labelpad=15)
plt.savefig(f"{clustering_config['files_dir']}/Krippendorff_dendrogram.png")

plt.figure(num=2, figsize=(20,10))
plot_dendrogram(model_agglomerative_kpca)
plt.title("Dendrogram Kernel PCA")
plt.tick_params(axis='both', which='major', labelsize=12)
plt.legend(('cluster_3', 'cluster_1', 'cluster_2', 'cluster_0'))
plt.xlabel("Unique annotator ID", fontsize=12, labelpad=15)
plt.ylabel("Node level", fontsize=12, labelpad=15)
plt.savefig(f"{clustering_config['files_dir']}/KPCA_dendrogram.png")


#create the training sets per cluster
traits = clustering_config["traits"]
parent = config["parent_text"]
seed = config["seed_data"]

DIR_krippendorff = f"./{clustering_config['methods'][0]}/"
DIR_kpca = f"./{clustering_config['methods'][1]}/"

df_krippendorff = generate_df(df, list_annotators, labels_krippendorff, clustering_config['methods'][0])
df_kpca = generate_df(df, list_annotators, labels_kpca, clustering_config['methods'][1])

df_selected_krippendorff, agg_train_set_krippendorff, agg_val_set_krippendorff = split_datasets(df_krippendorff, seed, DIR_krippendorff, traits, parent)
df_selected_kpca, agg_train_set_kpca, agg_val_set_kpca = split_datasets(df_kpca, seed, DIR_kpca, traits, parent)