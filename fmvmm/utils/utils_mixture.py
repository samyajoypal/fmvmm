import numpy as np
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt


def acc_check(tru, obs):
    lom = confusion_matrix(tru, obs).tolist()

    match = [max(lom[0])]
    max_prob_c = [lom[0].index(max(lom[0]))]
    for i in range(1, len(lom)):
        for j in max_prob_c:
            del lom[i][j]
        max_prob_c_temp = lom[i].index(max(lom[i]))
        max_prob_c.append(max_prob_c_temp)
        temp_match = max(lom[i])
        match.append(temp_match)

    match = np.sum([max(i) for i in lom])
    acc = match/len(tru)

    return acc


def prec_check(tru, obs):
    lom = confusion_matrix(tru, obs).tolist()
    lom_c = copy.deepcopy(lom)

    match = [max(lom_c[0])]
    max_prob_c = [lom[0].index(max(lom_c[0]))]
    for i in range(1, len(lom_c)):
        for j in sorted(max_prob_c, reverse=True):
            del lom_c[i][j]
        max_prob_c_temp = lom[i].index(max(lom_c[i]))
        max_prob_c.append(max_prob_c_temp)
        temp_match = max(lom_c[i])
        match.append(temp_match)
    lom = confusion_matrix(tru, obs).tolist()
    prec = []
    for k in range(len(match)):
        temp_prec = match[k]/np.sum(lom[k])
        prec.append(temp_prec)

    return np.mean(prec)


def rec_check(tru, obs):
    lom = confusion_matrix(tru, obs).tolist()
    lom_c = copy.deepcopy(lom)

    match = [max(lom_c[0])]
    max_prob_c = [lom_c[0].index(max(lom_c[0]))]
    for i in range(1, len(lom_c)):
        for j in sorted(max_prob_c, reverse=True):
            del lom_c[i][j]
        max_prob_c_temp = lom[i].index(max(lom_c[i]))
        max_prob_c.append(max_prob_c_temp)
        temp_match = max(lom_c[i])
        match.append(temp_match)

    lom = confusion_matrix(tru, obs).tolist()
    rec = []
    for k in range(len(match)):
        temp_rec = match[k]/np.sum([lom[l][max_prob_c[k]]
                                   for l in range(len(lom))])
        rec.append(temp_rec)

    return np.mean(rec)


def f_score(tru, obs):
    prec = prec_check(tru, obs)
    rec = rec_check(tru, obs)
    f_sc = 2/(prec**(-1)+rec**(-1))

    return f_sc


# def mixture_clusters(gamma_matrix, data_lol):
#     n = len(data_lol)
#     k = len(gamma_matrix[0])
#     cluster = []
#     data_cwise = [[] for i in range(k)]
#     for h in range(n):
#         max_prob_c = gamma_matrix[h].index(max(gamma_matrix[h]))
#         data_cwise[max_prob_c].append(data_lol[h])
#         cluster.append(max_prob_c)

#     return cluster, data_cwise

def mixture_clusters(gamma_matrix, data_lol):
    max_prob_c = np.argmax(gamma_matrix, axis=1)  # Get index of maximum probability for each row
    data_cwise = [[] for _ in range(gamma_matrix.shape[1])]  # Initialize list of lists

    for i, c in enumerate(max_prob_c):
        data_cwise[c].append(data_lol[i])

    return max_prob_c, data_cwise

def plot_correlation_heatmap(covariance_matrix, variable_names):
    # Calculate correlation matrix from the covariance matrix
    correlation_matrix = np.corrcoef(covariance_matrix)

    # Create the heatmap plot
    fig, ax = plt.subplots()
    im = ax.imshow(correlation_matrix, cmap='RdYlBu', vmin=-1, vmax=1)

    # Set x-axis and y-axis tick labels
    ax.set_xticks(np.arange(len(variable_names)))
    ax.set_yticks(np.arange(len(variable_names)))
    ax.set_xticklabels(variable_names)
    ax.set_yticklabels(variable_names)

    # Rotate x-axis tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Correlation')

    # Set plot title
    ax.set_title('Correlation Heatmap')

    # Show the plot
    plt.show()
    
    
def create_cluster_dataframes(dataframe, cluster_labels):
    """
    Create different dataframes for each cluster based on the provided labels.

    Parameters:
    - dataframe: pd.DataFrame, the original dataframe containing numerical data
    - cluster_labels: list, true cluster labels for each data point

    Returns:
    - cluster_dataframes: list, a list containing dataframes for each cluster
    """

    # Check if the length of cluster_labels matches the number of rows in the dataframe
    if len(cluster_labels) != len(dataframe):
        raise ValueError("Length of cluster_labels should match the number of rows in the dataframe.")

    # Create an empty list to store dataframes for each cluster
    cluster_dataframes = []

    # Iterate through unique cluster labels
    unique_clusters = set(cluster_labels)
    for cluster in unique_clusters:
        # Filter rows belonging to the current cluster
        cluster_data = dataframe[cluster_labels == cluster]

        # Store the dataframe in the dictionary with the cluster label as the key
        cluster_dataframes.append(cluster_data)

    return cluster_dataframes