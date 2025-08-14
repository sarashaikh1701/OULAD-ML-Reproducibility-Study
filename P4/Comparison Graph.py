#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# Labels for y-axis (Week + Model)
labels = [
    "W5 - LR", "W5 - SVM", "W5 - ANN", "W5 - LSTM",
    "W10 - LR", "W10 - SVM", "W10 - ANN", "W10 - LSTM",
    "W20 - LR", "W20 - SVM", "W20 - ANN", "W20 - LSTM",
    "W38 - LR", "W38 - SVM", "W38 - ANN", "W38 - LSTM"
]

# Paper accuracies
paper_accuracies = [
    0.70, 0.68, 0.74, 0.8082,
    0.75, 0.72, 0.79, 0.90,
    0.79, 0.70, 0.83, 0.94,
    0.81, 0.74, 0.85, 0.9523
]

# Reproduced accuracies
our_accuracies = [
    0.6812, 0.6947, 0.7021, 0.7176,
    0.7035, 0.7169, 0.7294, 0.8065,
    0.7812, 0.7926, 0.7980, 0.8433,
    0.8262, 0.8369, 0.8451, 0.8632
]

y = np.arange(len(labels))
height = 0.35

fig, ax = plt.subplots(figsize=(10, 8))

bars1 = ax.barh(y - height/2, paper_accuracies, height, label='Paper-reported', color='skyblue')
bars2 = ax.barh(y + height/2, our_accuracies, height, label='Reproduced', color='navy')

# Labels and formatting
ax.set_xlabel('Accuracy')
ax.set_title('Accuracy comparison by model and week', fontsize=14, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0.60, 1.00)
ax.legend(title="Source", fontsize=10, title_fontsize=11)

# Grid lines
ax.xaxis.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

