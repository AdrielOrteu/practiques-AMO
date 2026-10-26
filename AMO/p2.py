import numpy as np
import matplotlib.pyplot as plt

# 1. Define the sigmoid prediction probability (y_hat) from z
# z ranges from -10 to 10
z = np.linspace(-10, 10, 400)
y_hat = 1 / (1 + np.exp(-z))

# 2. Calculate costs for y = 1 and y = 0
cost_y1 = -np.log(y_hat)
cost_y0 = -np.log(1 - y_hat)

# 3. Plotting
plt.figure(figsize=(12, 5))

# Plot for y = 1
plt.subplot(1, 2, 1)
plt.plot(y_hat, cost_y1, color='blue', linewidth=2)
plt.title('Cost Function when True Label ($y$) = 1')
plt.xlabel('Predicted Probability ($\hat{y}$)')
plt.ylabel('Cost $J(\hat{y}, y)$')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 5)

# Plot for y = 0
plt.subplot(1, 2, 2)
plt.plot(y_hat, cost_y0, color='red', linewidth=2)
plt.title('Cost Function when True Label ($y$) = 0')
plt.xlabel('Predicted Probability ($\hat{y}$)')
plt.ylabel('Cost $J(\hat{y}, y)$')
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 5)

plt.tight_layout()
plt.show()