import math
from sklearn import datasets


def load_iris_dataset():
    """
    Loads Iris dataset
    :return: input data (x) and target labels (y)
    """

    iris_dataset = datasets.load_iris()
    x = iris_dataset.data
    y = iris_dataset.target
    return x, y


def split_data(data, label, split_ratio=0.8):
    """
    Splits the data into training and test sets
    :param data: input data as array
    :param label: target label as array
    :param split_ratio: Ratio to split Iris dataset
    :return: data/label splits as a dictionary.
    """

    dataset_length = len(label)
    split_pt = round(dataset_length * split_ratio)

    dataset_dict = {
        'train_data': data[0: split_pt],
        'test_data': data[split_pt: dataset_length],
        'train_label': label[0: split_pt],
        'test_label': label[split_pt: dataset_length]

    }
    print('Training set size: {} \nTesting set size: {}'.format(
        split_pt, dataset_length - split_pt))
    return dataset_dict


def compute_l2_distance(array_a, array_b):
    """
    Computes L2 norm.
    :param array_a: array of features
    :param array_b: array of features to compare with
    :return: l2 distance
    """

    pow_dist = 0
    for (elem_a, elem_b) in (zip(array_a, array_b)):
        pow_dist += math.pow(elem_a - elem_b, 2)
    dist = math.sqrt(pow_dist)
    return dist
