import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Classification import X, y
from RandomForest import rf_best_model
from load_and_process_data import data

importances = rf_best_model.feature_importances_

# Sort feature importances in descending order
sorted_indices = np.argsort(importances)[::-1]

# Cumulative sum of feature importances
cumulative_importances = np.cumsum(importances[sorted_indices])

# Find the number of features that account for 90% of prediction
num_features_90_percent = np.argmax(cumulative_importances >= 0.9) + 1

print("Of the 743 features, how many of these accounting for 90% of prediction:", num_features_90_percent)

# Plot the feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[sorted_indices], align='center', color='blue', label='Feature Importance')

# Highlight the features that account for 90% of prediction
plt.bar(range(num_features_90_percent), importances[sorted_indices][:num_features_90_percent],
        align='center', color='red', label='90% Prediction Importance')

plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.title('Feature Importance')
plt.legend()
plt.show()

# rank the features by importance
feature_order_by_importance = importances[sorted_indices]
feature_order_by_importance[:10]

# top 10 features that are important
X.columns[sorted_indices][:10]

df = data[X.columns[sorted_indices][:10]]
df['target'] = y
df.head(3)
class_0_data = df[df['target'] == "Small Cell"].drop(columns=['target'])
class_1_data = df[df['target'] == "Non-Small Cell"].drop(columns=['target'])

# Plot histograms for each feature comparing the two classes
for feature in df.columns[:-1]:  # Exclude the target column from iteration
    plt.figure(figsize=(7, 5))
    plt.hist(class_0_data[feature], bins=25, alpha=0.5, label='Small Cell', color='blue')
    plt.hist(class_1_data[feature], bins=25, alpha=0.5, label='Non-Small Cell', color='red')
    plt.title(f'Distribution of {feature} for Small Cell and Non-Small Cell Lung Cancer')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    print()

miRNA_data = {
    'MIMAT0000078' : 'hsa-miR-23a-3p',
    'MIMAT0000772' : 'hsa-miR-345-5p',
    'MIMAT0000098' : 'hsa-miR-100-5p',
    'MIMAT0000100': 'hsa-miR-375-3p',
    'MIMAT0000084' : 'hsa-mir-27a',
    'MIMAT0000086' : 'hsa-miR-29a-3p',
    'MIMAT0000279' : 'hsa-miR-222-3p',
    'MIMAT0000728': 'hsa-miR-375-3p',
    'MIMAT0002807' : 'hsa-miR-29a-3p',
    'MIMAT0000428' : 'hsa-miR-135a-5p',
}

table = pd.DataFrame(list(miRNA_data.items()), columns=['miRNA Identifier', 'miRNA'])

