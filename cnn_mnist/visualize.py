import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Your confusion matrix from the output
confusion_matrix = np.array([
    [ 975,    1,    0,    1,    0,    0,    2,    0,    1,    0],
    [   0, 1132,    0,    1,    0,    0,    0,    1,    1,    0],
    [   1,    0, 1027,    0,    0,    0,    0,    3,    1,    0],
    [   0,    0,    2, 1002,    0,    3,    0,    0,    3,    0],
    [   0,    0,    0,    0,  980,    0,    0,    1,    0,    1],
    [   2,    0,    0,    5,    0,  882,    2,    1,    0,    0],
    [   1,    2,    1,    0,    1,    1,  949,    0,    3,    0],
    [   0,    1,    5,    1,    0,    0,    0, 1019,    1,    1],
    [   2,    0,    2,    0,    0,    1,    0,    0,  968,    1],
    [   2,    1,    0,    0,   11,    5,    0,    5,    6,  979]]
)

plt.figure(figsize=(12, 10))

# Create subplots for regular and log scale
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# For better visualization of log scale, need to replace zeros with a very small number
# (log(0) is undefined)
confusion_matrix_log = confusion_matrix.copy()
confusion_matrix_log[confusion_matrix_log == 0] = 0.1  # Small value instead of zero

# Plot regular scale confusion matrix
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_title('Confusion Matrix - Regular Scale', fontsize=16)
ax1.set_xlabel('Predicted Label', fontsize=14)
ax1.set_ylabel('True Label', fontsize=14)

# Plot log scale confusion matrix
sns.heatmap(confusion_matrix_log, annot=True, fmt=".1f", cmap="Blues", norm=LogNorm(), ax=ax2)
ax2.set_title('Confusion Matrix - Logarithmic Scale', fontsize=16)
ax2.set_xlabel('Predicted Label', fontsize=14)
ax2.set_ylabel('True Label', fontsize=14)

plt.tight_layout()

# Create a figure for metrics visualization
plt.figure(figsize=(14, 6))

# Extract metrics from your output
classes = range(10)
precision_values = [0.99, 1.00, 0.99, 0.99, 0.99, 0.99, 1.00, 0.99, 0.98, 1.00]
recall_values = [0.99, 1.00, 1.00, 0.99, 1.00, 0.99, 0.99, 0.99, 0.99, 0.97]
f1_values = [0.99, 1.00, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.98]

# Create bar plot for precision, recall, and f1-score for each class
x = np.arange(len(classes))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width, precision_values, width, label='Precision')
rects2 = ax.bar(x, recall_values, width, label='Recall (Sensitivity)')
rects3 = ax.bar(x + width, f1_values, width, label='F1-score')

ax.set_title('Precision, Recall and F1-score by Class', fontsize=16)
ax.set_xlabel('Class', fontsize=14)
ax.set_ylabel('Score', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.7)

# Add text annotations on the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.tight_layout()
plt.show()
plt.savefig("dupa.png")
