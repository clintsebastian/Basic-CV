# Iris Dataset from http://archive.ics.uci.edu/ml/datasets/Iris

import numpy as np
from scipy import stats
from basics.common import load_iris_dataset, split_data, compute_l2_distance


def get_top_k_neighbors(distances_list, k):
    """
    Collects top k neighbors of a query
    :param distances_list: list of computed distances
    :param k: number of neighbors to consider
    :return: returns top-k ranks
    """
    # Note that from here, we transition to numpy arrays instead of lists
    distance_array = np.array(distances_list)
    top_k_ranks = np.argsort(distance_array)
    return top_k_ranks[: k]


def compute_full_distance(test_sample, train_dataset):
    """
    Computes distance of each test sample to every training sample.
    :param test_sample:
    :param train_dataset:
    :return: list of distance
    """
    test_sample_distances = list([])
    for train_sample in train_dataset:
        sample_distance = compute_l2_distance(test_sample, train_sample)
        test_sample_distances.append(sample_distance)
    return test_sample_distances


def apply_knn(train_dataset, test_dataset, k=1):
    """
    Applies knn across the entire test set.
    :param train_dataset:
    :param test_dataset:
    :param k: number of neighbors to consider
    :return:
    """
    nn_array = np.zeros(shape=(test_dataset.shape[0], k))
    for i, test_sample in enumerate(test_dataset):
        distance_list = compute_full_distance(test_sample, train_dataset)
        k_nn = get_top_k_neighbors(distance_list, k)
        nn_array[i] = k_nn
    return nn_array.astype(int)


def compute_accuracy(nn_array, train_label, test_label):
    """
    Computes accuracy
    :param nn_array: nearest neighbors of each element of test set
    :param train_label: array of training labels
    :param test_label: array of test labels
    :return: accuracy value between 0 and 1
    """
    correct_count = 0
    for test_idx, train_idx in enumerate(nn_array):
        prediction = train_label[train_idx]
        target = test_label[test_idx]
        # Take the most recurring label out of the predictions.
        if stats.mode(prediction)[0] == target:
            correct_count += 1
    return correct_count/len(test_label)


if __name__ == '__main__':
    x, y = load_iris_dataset()
    dataset = split_data(data=x, label=y, split_ratio=0.8)

    nearest_neighbors = apply_knn(train_dataset=dataset['train_data'],
                                  test_dataset=dataset['test_data'], k=1)

    accuracy = compute_accuracy(nn_array=nearest_neighbors,
                                train_label=dataset['train_label'],
                                test_label=dataset['test_label'])
    print('Classification accuracy: {}'.format(round(accuracy * 100, 2)))

