import numpy as np
from sklearn.metrics import confusion_matrix
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from scipy.optimize import linear_sum_assignment

def sample_mixture_distribution(N, rand_func, pis, alphas, mixture_type = "identical", random_state=None):
    """
    Generate N samples from a mixture of a given probability distribution,
    along with the corresponding component labels.

    Parameters
    ----------
    N : int
        Total number of samples.
    rand_func : callable (identical case) or list of callable (non-identical case)
        A function that generates random samples from a given distribution.
        It should take parameters in 'alphas[k]' and return a (M, p) array.
    pis : list or array, shape (K,)
        Mixture proportions (must sum to 1).
    alphas : list of length K
        List of parameter sets for each component, where each alphas[k]
        is a dictionary or list of parameters to be passed to `rand_func`.
    random_state : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    X : array, shape (N, p)
        Random samples from the mixture distribution.
    labels : array, shape (N,)
        Integer labels (0,1,...,K-1) indicating which component each sample belongs to.
    """
    rng = np.random.default_rng(random_state)
    K = len(pis)
    if mixture_type == "identical":
        assert K==len(rand_func)
    # Compute number of samples for each component (except last)
    sample_counts = [int(np.floor(N * pi)) for pi in pis[:-1]]

    # Ensure total adds up to N (assign remaining to last component)
    sample_counts.append(N - sum(sample_counts))

    # Generate samples from each component and store labels
    samples = []
    labels = []
    for k in range(K):
        M_k = sample_counts[k]  # Number of samples for this component
        if M_k > 0:
            if mixture_type != "identical":
                samples.append(rand_func(*alphas[k], *[M_k]))  # Call the generator with params
            else:
                samples.append(rand_func[k](*alphas[k], *[M_k]))
            labels.extend([k] * M_k)  # Assign component label k to these samples

    # Concatenate samples and labels
    X = np.vstack(samples)
    labels = np.array(labels)

    # Shuffle samples and labels together to avoid ordering artifacts
    indices = np.arange(N)
    rng.shuffle(indices)
    X = X[indices]
    labels = labels[indices]

    return X, labels


def map_labels(true_labels, predicted_labels):
    """
    Maps predicted cluster labels to the true labels using the Hungarian algorithm
    to resolve label switching.

    Parameters:
    - true_labels: array-like, shape (n_samples,), ground-truth labels.
    - predicted_labels: array-like, shape (n_samples,), predicted labels.

    Returns:
    - mapped_predicted_labels: np.ndarray, mapped predicted labels.
    """
    true_labels, predicted_labels = np.array(true_labels), np.array(predicted_labels)

    unique_true_labels = np.unique(true_labels)
    unique_pred_labels = np.unique(predicted_labels)

    cost_matrix = np.zeros((len(unique_true_labels), len(unique_pred_labels)))

    for i, true_label in enumerate(unique_true_labels):
        for j, pred_label in enumerate(unique_pred_labels):
            match = (true_labels == true_label).astype(int)
            pred = (predicted_labels == pred_label).astype(int)
            cost_matrix[i, j] = -f1_score(match, pred)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    label_mapping = {unique_pred_labels[col]: unique_true_labels[row] for row, col in zip(row_ind, col_ind)}
    mapped_predicted_labels = np.vectorize(label_mapping.get)(predicted_labels)

    return mapped_predicted_labels


def clustering_metrics(true_labels, predicted_labels):
    """
    Computes Accuracy, Precision, Recall, and F-score for clustering with label switching handled.

    Parameters:
    - true_labels: array-like, shape (n_samples,), ground-truth labels.
    - predicted_labels: array-like, shape (n_samples,), predicted labels.

    Returns:
    - metrics: dict containing Accuracy, Precision, Recall, and F-score.
    """
    mapped_preds = map_labels(true_labels, predicted_labels)

    metrics = {
        "accuracy": accuracy_score(true_labels, mapped_preds),
        "precision": precision_score(true_labels, mapped_preds, average='weighted', zero_division=0),
        "recall": recall_score(true_labels, mapped_preds, average='weighted', zero_division=0),
        "f_score": f1_score(true_labels, mapped_preds, average='weighted')
    }

    return metrics










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
