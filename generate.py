import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = 2 * X + np.random.randn(100) * 2  # y = 2 * x + noise

# Save as CSV
data = pd.DataFrame({'X': X, 'y': y})
data.to_csv('data/synthetic_data.csv', index=False)

# Plot the data
plt.scatter(X, y)
plt.title('Synthetic Data for Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
