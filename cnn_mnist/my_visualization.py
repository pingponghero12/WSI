import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

# Your confusion matrix from the output
confusion_matrix = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 2],
    [0, 2, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 2, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 2],
    [0, 0, 0, 0, 0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 2]
])

# Extract metrics from the classification report
classes = range(10)
precision_values = [1.00, 1.00, 1.00, 0.40, 1.00, 0.60, 0.00, 0.60, 0.50, 0.25]
recall_values = [0.33, 0.67, 0.33, 0.67, 0.33, 1.00, 0.00, 1.00, 0.33, 0.67]
f1_values = [0.50, 0.80, 0.50, 0.50, 0.50, 0.75, 0.00, 0.75, 0.40, 0.36]

plt.figure(figsize=(12, 10))

# Create subplots for regular and log scale confusion matrices
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# For log scale, replace zeros with a very small number (log(0) is undefined)
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
fig, ax = plt.subplots(figsize=(14, 8))

# Create bar plot for precision, recall, and f1-score for each class
x = np.arange(len(classes))
width = 0.25

rects1 = ax.bar(x - width, precision_values, width, label='Precision')
rects2 = ax.bar(x, recall_values, width, label='Recall (Sensitivity)')
rects3 = ax.bar(x + width, f1_values, width, label='F1-score')

ax.set_title('Precision, Recall and F1-score by Class', fontsize=16)
ax.set_xlabel('Class', fontsize=14)
ax.set_ylabel('Score', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(classes)
ax.set_ylim(0, 1.1)  # Set y-axis limit from 0 to 1.1 to show full bars
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
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Add a summary metrics visualization
fig_summary, ax_summary = plt.subplots(figsize=(10, 6))

# Overall metrics
metrics_names = ['Accuracy', 'Precision (Macro Avg)', 'Recall (Macro Avg)', 'F1-Score (Macro Avg)']
metrics_values = [0.5333, 0.635, 0.53, 0.51]  # From your evaluation results

# Create horizontal bar chart for overall metrics
bars = ax_summary.barh(metrics_names, metrics_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax_summary.set_xlim(0, 1)
ax_summary.set_title('Overall Model Performance Metrics', fontsize=16)
ax_summary.set_xlabel('Score', fontsize=14)
ax_summary.grid(True, linestyle='--', alpha=0.7, axis='x')

# Add value labels to the bars
for bar in bars:
    width = bar.get_width()
    ax_summary.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
            ha='left', va='center', fontsize=12)

plt.tight_layout()

# Add a timestamp and user info as a footer
fig.text(0.5, 0.01, f"Generated for user: pingponghero12 | Date: 2025-03-28 12:44:25 UTC", 
         ha='center', fontsize=10, style='italic')

plt.show()
