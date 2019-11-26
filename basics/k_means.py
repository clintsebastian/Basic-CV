import numpy as np
import matplotlib.pyplot as plt
from basics.common import compute_l2_distance, load_iris_dataset, split_data


def get_iris_feature(feat_x_id=2, feat_y_id=3):
    """
    Iris dataset has 4 features.
    1. Petal length
    2. Petal width
    3. Sepal length
    4. Sepal width

    :param feat_x_id: select id of feature x
    :param feat_y_id: select id of feature y
    :return: feature x, y and corresponding label
    """
    if 0 <= feat_x_id < 4 or 0 <= feat_y_id < 4:
        pass
    else:
        raise ValueError('Incorrect feature, value must be between 0 and 4')
    x, y = load_iris_dataset()
    dataset = split_data(x, y, 1.0)
    train_data = dataset['train_data']
    feature_x = train_data[:, feat_x_id]
    feature_y = train_data[:, feat_y_id]
    label = dataset['train_label']

    return feature_x, feature_y, label


def sample_centroids(n_clusters, x_min=0, y_min=0, x_max=100, y_max=100):
    """
    Generate centroid for each cluster.
    Initializing x and y in a reasonable range can yield easier convergence
    :param n_clusters: Number of cluster centroids to generate
    :param x_min: min value of x coordinate of centroid
    :param y_min: min value of y coordinate of centroid
    :param x_max: max value of x coordinate of centroid
    :param y_max: max value of y coordinate of centroid
    :return: initial cluster centroids
    """
    cluster_centroids = np.zeros(shape=(n_clusters, 2))
    for i in range(n_clusters):
        np.random.seed(i)
        x = np.random.randint(x_min, x_max)
        y = np.random.randint(y_min, y_max)
        cluster_centroids[i, :] = np.array([x, y])
    return cluster_centroids


def initialize_centroids(n_clusters, feat_x, feat_y):
    """
    Initializes centroids from the min and max of features x and y
    :return: cluster centers
    """
    cluster_centers = sample_centroids(n_clusters=n_clusters,
                                       x_min=np.min(feat_x),
                                       y_min=np.min(feat_y),
                                       x_max=np.max(feat_x),
                                       y_max=np.max(feat_y))
    return cluster_centers


def update_centroids(assigned_label, feat_positions, n_clusters):
    """
    Updates the centroids for K-means
    :param assigned_label: Assigned label
    :param feat_positions: Features x and y (could be visualized in 2D space)
    :param n_clusters: number of clusters
    :return: updated centroids
    """

    updated_centroids = np.zeros(shape=(n_clusters, 2))

    for i in range(n_clusters):
        feat_per_label = feat_positions[assigned_label == i]
        updated_centroids[i, :] = np.mean(feat_per_label, axis=0)
    return updated_centroids


def assignment_step(feat_x, feat_y, label, cluster_centers, n_clusters):
    """
    Assigns cluster labels to the after computing distances to the centroids.
    :return: assigned labels for each sample, features
    """
    sample_distances = np.zeros(shape=(len(label), n_clusters))
    feats = np.zeros(shape=(len(label), 2))

    for i, sample in enumerate(zip(feat_x, feat_y)):
        for j, centroid in enumerate(cluster_centers):
            per_sample_distance = compute_l2_distance(sample, centroid)
            sample_distances[i, j] = per_sample_distance
            feats[i, :] = sample

    # Select the label corresponding least distance
    assigned_labels = np.argmin(sample_distances, axis=1)
    return assigned_labels, feats


def visualize_features(feats_x, feats_y, labels, c_map, marker='.', size=50):
    for x, y, l in zip(feats_x, feats_y, labels):
        plt.scatter(x, y, color=c_map[l], marker=marker, s=size)


def create_color_map(n_clusters):
    """ Randomly assigns colors to each cluster """
    c_map = dict({})
    for i in range(n_clusters):
        c_map[i] = np.random.rand(3,)
    return c_map


def k_means(n_clusters=4, iterations=1000):
    """
    Apply k means clustering
    :param n_clusters: number of clusters
    :param iterations: number of iterations run K-means algorithm
    :return:
    """
    c_map = create_color_map(n_clusters)
    plt.subplot()

    feats_x, feats_y, labels = get_iris_feature()
    centers = initialize_centroids(n_clusters, feats_x, feats_y)

    # Randomly initialize assign labels
    assign_labels = np.random.randint(n_clusters, size=len(labels))

    for i in range(iterations):

        prev_labels = assign_labels
        assign_labels, feats = assignment_step(feats_x, feats_y, labels, centers, n_clusters)
        visualize_features(feats[:, 0], feats[:, 1], assign_labels, c_map)
        centers = update_centroids(assign_labels, feats, n_clusters)
        visualize_features(centers[:, 0], centers[:, 1], np.unique(labels), c_map, marker='X', size=100)
        plt.show()

        # Comparing previous and current labels
        if (prev_labels == assign_labels).all():
            print('Converged at iteration at {}'.format(i))
            break
        else:
            print('Iteration {}'.format(i))


if __name__ == '__main__':
    k_means(n_clusters=3, iterations=50)
