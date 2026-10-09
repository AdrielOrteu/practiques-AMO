import math
import random
import matplotlib.pyplot as plt

### Logistic Regression
def generate_dataset_1(n: int) -> dict[float, float]:
    """Generates a dataset of `n` unique floating-point features (x)

    mapped to binary targets y (0.0 or 1.0) using a sigmoid curve.
    """
    dataset = {}
    while len(dataset) < n:
        # Generate a continuous feature x
        x = round(random.uniform(-3.0, 3.0), 4)

        # Calculate logistic probability: P(y=1|x) = 1 / (1 + e^-x)
        prob = 1 / (1 + math.exp(-1.5 * x))

        # Sample binary label (0.0 or 1.0) based on probability
        y = 1.0 if random.random() < prob else 0.0

        dataset[x] = y

    return dataset

def generate_dataset_2(n: int) -> dict[tuple[float, float], float]:
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

# Example usage:
# data = generate_dataset(5)
# print(data)
# Output: {-2.1402: 0.0, 0.4512: 1.0, -0.8321: 0.0, 1.9104: 1.0, 0.1293: 1.0}
def show_data_set_1():
    # Generate dataset with 100 sample points
    dataset = generate_dataset_1(100)
    x_pts = list(dataset.keys())
    y_pts = list(dataset.values())
    
    # Theoretical sigmoid probability curve for reference: P(y=1|x) = 1 / (1 + e^-1.5x)
    x_line = [i / 100 for i in range(-300, 301)]
    y_line = [1 / (1 + math.exp(-1.5 * x)) for x in x_line]
    
    # Plotting
    plt.figure(figsize=(8, 5))
    plt.scatter(
        x_pts,
        y_pts,
        c=y_pts,
        cmap="coolwarm",
        edgecolors="k",
        alpha=0.8,
        zorder=3,
        label="Generated Samples ($y \\in \\{0, 1\\}$)",
    )
    plt.plot(
        x_line,
        y_line,
        color="black",
        linestyle="--",
        linewidth=2,
        label="True Probability $P(y=1|x)$",
    )
    
    plt.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="Threshold (0.5)")
    plt.title("Synthetic Logistic Regression Dataset ($n = 100$)", fontsize=12)
    plt.xlabel("Feature ($x$)")
    plt.ylabel("Target Class / Probability ($y$)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()
# Generate dataset with 100 sample points
dataset = generate_dataset_2(100)

# Separate points by class label
x0 = [pt[0] for pt, label in dataset.items() if label == 0.0]
y0 = [pt[1] for pt, label in dataset.items() if label == 0.0]

x1 = [pt[0] for pt, label in dataset.items() if label == 1.0]
y1 = [pt[1] for pt, label in dataset.items() if label == 1.0]

# Plotting the 2D clouds
plt.figure(figsize=(8, 6))
plt.scatter(
    x0, y0, color="crimson", label="Class 0 (0.0)", edgecolors="k", alpha=0.8
)
plt.scatter(
    x1,
    y1,
    color="royalblue",
    label="Class 1 (1.0)",
    edgecolors="k",
    alpha=0.8,
)

plt.axhline(0, color="gray", linestyle=":", alpha=0.5)
plt.axvline(0, color="gray", linestyle=":", alpha=0.5)

plt.title("Synthetic 2D Point Clouds ($n = 100$)", fontsize=12)
plt.xlabel("Feature $x_1$")
plt.ylabel("Feature $x_2$")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()