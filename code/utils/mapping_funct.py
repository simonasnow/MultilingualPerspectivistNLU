import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def mapping_nationality (df_nationality):
    list_nationality = list(df_nationality['Nationality'])
    nationality_ground_truth = list()
    colors_nationality = list()
    for el in list_nationality:
        if el == 'United Kingdom':
            nationality_ground_truth.append(0)
            colors_nationality.append('r')
        elif el == 'United States':
            nationality_ground_truth.append(1)
            colors_nationality.append('b')
        elif el == 'India':
            nationality_ground_truth.append(2)
            colors_nationality.append('g')
        elif el == 'Australia':
            nationality_ground_truth.append(3)
            colors_nationality.append('m')
        elif el == 'Ireland':
            nationality_ground_truth.append(4)
            colors_nationality.append('y')
    #print(nationality_ground_truth)

    return nationality_ground_truth, colors_nationality

def mapping_gender (df_gender):
    list_gender = list(df_gender['Gender'])
    gender_ground_truth = list()
    colors_gender = list()
    for el in list_gender:
        if el == 'Male':
            gender_ground_truth.append(0)
            colors_gender.append('b')
        elif el == 'Female':
            gender_ground_truth.append(1)
            colors_gender.append('r')

    return gender_ground_truth, colors_gender

def mapping_generation (df_generation):
    list_generation = list(df_generation['Generation'])
    generation_ground_truth = list()
    colors_generation = list()
    for el in list_generation:
        if el == 'boomer':
            generation_ground_truth.append(0)
            colors_generation.append('r')
        elif el == 'gen y':
            generation_ground_truth.append(1)
            colors_generation.append('b')
        elif el == 'gen x':
            generation_ground_truth.append(2)
            colors_generation.append('g')
        elif el == 'gen z':
            generation_ground_truth.append(3)
            colors_generation.append('m')
        else:
            generation_ground_truth.append(4)
            colors_generation.append('y')

    return generation_ground_truth, colors_generation

def intrinsic_evaluation (model, distance_matrix, labels):
    print ("Dendrogram sklearn estimated number of clusters: ", model.n_clusters_)
    print(f"Silhouette Coefficient: {metrics.silhouette_score(distance_matrix, labels):.3f}")
    print(f"Calinski-Harabasz Index: {metrics.calinski_harabasz_score(distance_matrix, labels):.3f}")
    print(f"Davies-Bouldin Index: {metrics.davies_bouldin_score(distance_matrix, labels):.3f}")


def mapping_evaluation (gender_ground_truth, nationality_ground_truth, generation_ground_truth, model_labels):
    print ("\n", "Gender")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(gender_ground_truth, model_labels):.3f}")
    print(f"Adjusted Mutual Information:{metrics.adjusted_mutual_info_score(gender_ground_truth, model_labels):.3f}")
    print ("\n","Nationality")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(nationality_ground_truth, model_labels):.3f}")
    print(f"Adjusted Mutual Information:{metrics.adjusted_mutual_info_score(nationality_ground_truth, model_labels):.3f}")
    print ("\n","Generation")
    print(f"Adjusted Rand Index: {metrics.adjusted_rand_score(generation_ground_truth, model_labels):.3f}")
    print(f"Adjusted Mutual Information:{metrics.adjusted_mutual_info_score(generation_ground_truth, model_labels):.3f}")
    print()


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

def plot_mapped_data (model, labels_color):
    plt.figure(num=1, figsize=(16,10))
    plt.title("Mapping nationality (KPCA)")
    plot_dendrogram(model)
    plt.xlabel("Annotators")
    ax_n = plt.gca()
    x_lbls_n = ax_n.get_xmajorticklabels()
    for lbl in x_lbls_n:
        lbl.set_color(labels_color[lbl.get_text()])
    plt.show()