import pandas as pd
import numpy as np
import os
import pickle

from scipy.cluster.hierarchy import ClusterWarning
from warnings import simplefilter
simplefilter("ignore", ClusterWarning)

from config import config, clustering_config

from utils.mapping_funct import mapping_nationality, mapping_gender, mapping_generation, plot_mapped_data, mapping_evaluation, intrinsic_evaluation
from utils.hierarchical_clustering import agglomerative_clustering

# read data
df = pd.read_csv(f"{config['corpus_file']}")
df_demographics = df[['id_annotator','Gender', 'Nationality','Generation']].drop_duplicates()

distance_matrix_krippendorff = np.load(os.path.join(f"{clustering_config['files_dir']}", "distance_matrix_krippendorff.npy"))
distance_matrix_kpca = np.load(os.path.join(f"{clustering_config['files_dir']}", "distance_matrix_kpca.npy"))

with open(os.path.join(f"{clustering_config['files_dir']}", "list_annotators.pickle"), "rb") as handle:
    list_annotators=pickle.load(handle)


# krippendorff and kpca clusters
model_agglomerative_krippendorff = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["Krippendorff_distance_threshold"], distance_matrix_krippendorff)
labels_krippendorff = model_agglomerative_krippendorff.labels_

model_agglomerative_kpca = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["KPCA_distance_threshold"], distance_matrix_kpca)
labels_kpca = model_agglomerative_kpca.labels_

#intrinsic evaluation
print("Intrinsic evaluatoin Krippendorff:")
intrinsic_evaluation(model_agglomerative_krippendorff, distance_matrix_krippendorff, labels_krippendorff)
print("\n\n Intrinsic evaluation KPCA:")
intrinsic_evaluation(model_agglomerative_kpca, distance_matrix_kpca, labels_kpca)

# mapping nationality
# print ("\n\n"+"MAPPING NATIONALITY: ")
df_nationality = df_demographics[['id_annotator', 'Nationality']]
df_nationality = df_nationality.set_index('id_annotator')
df_nationality = df_nationality.loc[list_annotators]
nationality_ground_truth, colors_nationality = mapping_nationality(df_nationality)

# print("Krippendorff:")
model_krippendorff_nationality = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["krippendorff_nationality"], distance_matrix_krippendorff)
labels_krippendorff_nationality = model_krippendorff_nationality.labels_

# print("KPCA:")
model_kpca_nationality = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["kpca_nationality"], distance_matrix_kpca)
labels_kpca_nationality = model_kpca_nationality.labels_

labels_color_nationality = dict(zip(list(range(len(list_annotators))), colors_nationality))
labels_color_nationality = {str(k): str(v) for k,v in labels_color_nationality.items()}
plot_mapped_data(model_krippendorff_nationality, labels_color_nationality)
plot_mapped_data(model_kpca_nationality, labels_color_nationality)

# mapping gender
# print ("\n\n"+"MAPPING GENDER: ")
df_gender = df_demographics[['id_annotator', 'Gender']]
df_gender = df_gender.set_index('id_annotator')
df_gender = df_gender.loc[list_annotators]
gender_ground_truth, colors_gender = mapping_gender(df_gender)

# print("Krippendorff:")
model_krippendorff_gender = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["krippendorff_gender"], distance_matrix_krippendorff)
labels_krippendorff_gender = model_krippendorff_gender.labels_

# print("KPCA:")
model_kpca_gender = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["kpca_gender"], distance_matrix_kpca)
labels_kpca_gender = model_kpca_gender.labels_

labels_color_gender = dict(zip(list(range(len(list_annotators))), colors_gender))
labels_color_gender = {str(k): str(v) for k,v in labels_color_gender.items()}
plot_mapped_data(model_krippendorff_gender, labels_color_gender)
plot_mapped_data(model_kpca_gender, labels_color_gender)

# mapping generation
# print ("\n\n"+"MAPPING GENERATION: ")
df_generation = df_demographics[['id_annotator', 'Generation']]
df_generation = df_generation.set_index('id_annotator')
df_generation = df_generation.loc[list_annotators]
generation_ground_truth, colors_generation = mapping_generation(df_generation)

# print("Krippendorff:")
model_krippendorff_generation = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["krippendorff_generation"], distance_matrix_krippendorff)
labels_krippendorff_generation = model_krippendorff_generation.labels_

# print("KPCA:")
model_kpca_generation = agglomerative_clustering(clustering_config["linkage_method"], clustering_config["kpca_generation"], distance_matrix_kpca)
labels_kpca_generation = model_kpca_generation.labels_

labels_color_generation = dict(zip(list(range(len(list_annotators))), colors_generation))
labels_color_generation = {str(k): str(v) for k,v in labels_color_generation.items()}
plot_mapped_data(model_krippendorff_generation, labels_color_generation)
plot_mapped_data(model_kpca_generation, labels_color_generation)

if list_annotators == list(df_nationality.index) == list(df_gender.index) == list(df_generation.index):
    print ("\n\n"+"the lists are identical :)")
else:
    print("\n\n"+"the lists are different :(")


#intrinsic evaluation
print ("\nKrippendorff mapping with evaluation")
print ("number of clusters: ", model_agglomerative_krippendorff.n_clusters_)
mapping_evaluation(gender_ground_truth, nationality_ground_truth, generation_ground_truth, labels_krippendorff)

print ("\nKPCA mapping with evaluation")
print ("number of clusters: ", model_agglomerative_kpca.n_clusters_)
mapping_evaluation(gender_ground_truth, nationality_ground_truth, generation_ground_truth, labels_kpca)