from random import uniform, gauss
import matplotlib.pyplot as plt
import numpy as np

def noisy_cubic_distribution(a, b ,c, d, min, max , n:int)->list:
    sample = []
    for i in range(n):
        x = uniform(min, max)
        y = a*(x**3)+b*(x**2)+c*x+d
        y += gauss(0,0.25)
        sample.append((x,y))
    return sample

def plot_noisy_cubic(sample: list, true_cubic: tuple = (-1, 2, -1, 1)):
  """Plots a given list of noisy (x, y) datapoints against the true cubic

  function defined by coefficients (a, b, c, d).
  """
  # Extract coordinates from the provided sample
  x_vals = [pt[0] for pt in sample]
  y_vals = [pt[1] for pt in sample]

  # Determine x range dynamically from the sample data with a small padding
  x_min, x_max = min(x_vals), max(x_vals)
  padding = (x_max - x_min) * 0.1 if x_max != x_min else 0.5
  x_smooth = np.linspace(x_min - padding, x_max + padding, 200)

  # Calculate the true smooth curve
  a, b, c, d = true_cubic
  y_smooth = a * (x_smooth**3) + b * (x_smooth**2) + c * x_smooth + d

  # Create the visualization
  plt.figure(figsize=(9, 6))
  plt.plot(
      x_smooth,
      y_smooth,
      label=f"True Function: $y = {a}x^3 + {b}x^2 + {c}x + {d}$",
      color="royalblue",
      linewidth=2.5,
  )
  plt.scatter(
      x_vals,
      y_vals,
      color="crimson",
      label=f"Noisy Datapoints ($n={len(sample)}$)",
      zorder=5,
      s=50,
      edgecolor="black",
      linewidth=0.5,
  )

  plt.title(
      "True Cubic Function vs. Noisy Sample Datapoints",
      fontsize=14,
      fontweight="bold",
      pad=12,
  )
  plt.xlabel("X", fontsize=11)
  plt.ylabel("Y", fontsize=11)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(fontsize=10)
  plt.tight_layout()
  plt.show()

true_cubic = (-1,2,0,-0.5)
noisy_sample = noisy_cubic_distribution(
    true_cubic[0],
    true_cubic[1],
    true_cubic[2],
    true_cubic[3],
    0,
    2,
    20
)
plot_noisy_cubic(noisy_sample, true_cubic)

