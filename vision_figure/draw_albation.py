import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Data
shape_dimensions = [80, 40, 20, 10, 5, 0]
texture_dimensions = [80, 40, 20, 10, 5, 0]

# PC values based on the data provided
pc_values = [
    [0.9072, 0.9064, 0.9052, 0.9038, 0.9032, 0.9009],
    [0.9061, 0.9054, 0.8995, 0.8989, 0.8963, 0.8921],
    [0.9061, 0.9057, 0.8982, 0.8922, 0.8870, 0.8816],
    [0.9039, 0.9025, 0.8859, 0.8686, 0.8571, 0.8348],
    [0.9042, 0.9000, 0.8673, 0.8426, 0.8269, 0.7785],
    [0.9018, 0.8932, 0.8400, 0.7398, 0.6659, 0]
]

# Creating meshgrid for X and Y axes
x = np.array(shape_dimensions)
y = np.array(texture_dimensions)
x, y = np.meshgrid(x, y)

# Flatten the meshgrid and corresponding PC values for 3D plotting
x = x.flatten()
y = y.flatten()
pc_values_flattened = np.array(pc_values).flatten()

# Plotting
fig = plt.figure(figsize=(12, 10))  # Increased figure size
ax = fig.add_subplot(111, projection='3d')

# Color bars based on row (for example, row-wise coloring)
color_map = plt.cm.viridis  # You can change this colormap
colors = color_map(pc_values_flattened / np.max(pc_values_flattened))  # Normalize and apply colormap

# Creating a 3D bar plot with tightly arranged bars and custom colors
ax.bar3d(x, y, np.zeros_like(pc_values_flattened), 2, 2, pc_values_flattened,
    color=colors, edgecolor='black', shade=True)

# Customizing axis labels and title
ax.set_xlabel('Shape Dimension (λ₁)')
ax.set_ylabel('Texture Dimension (λ₂)')
ax.set_zlabel('PC Value')

# Title
ax.set_title('Dimension Ablation Results of Shape and Texture Coefficients')

# Customizing ticks for better clarity
ax.set_xticks(range(len(shape_dimensions)))
ax.set_xticklabels([str(val) for val in shape_dimensions])

ax.set_yticks(range(len(texture_dimensions)))
ax.set_yticklabels([str(val) for val in texture_dimensions])

# Setting Z-axis limit (adjust as needed)
ax.set_zlim(0.6, 0.92)

# Adjust layout to avoid overlap
plt.tight_layout()

# Manually adjust margins to ensure everything fits
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.savefig('3d_bar_plot_PC.png')
plt.show()
