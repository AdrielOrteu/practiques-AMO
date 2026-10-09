import math
import random

def generate_dataset(n: int) -> dict[tuple[float, float], float]:
    """Generates a 2D dataset with two distinct point clouds for logistic regression.

    Keys are (x1, x2) feature coordinates, values are binary labels (0.0 or
    1.0).
    """
    dataset = {}
    n_class0 = n // 2
    n_class1 = n - n_class0

    # Cluster 0: Centered around (-1.5, -1.5)
    while len([k for k, v in dataset.items() if v == 0.0]) < n_class0:
        x1 = round(random.gauss(-1.5, 0.75), 4)
        x2 = round(random.gauss(-1.5, 0.75), 4)
        dataset[(x1, x2)] = 0.0

    # Cluster 1: Centered around (1.5, 1.5)
    while len(dataset) < n:
        x1 = round(random.gauss(1.5, 0.75), 4)
        x2 = round(random.gauss(1.5, 0.75), 4)
        dataset[(x1, x2)] = 1.0

    return dataset

weights = [random.uniform(-1,1)*10, random.uniform(-1,1)*10]

def train(weights, dataset):


def cost(data, weights: list):
    return sum()