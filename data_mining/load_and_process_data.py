import pandas as pd
import numpy as np

filePath ="miRNA_lung.csv"
data = pd.read_csv(filePath)
data.head(3)
data['lineage'] = np.where(data['lineage_2'].str.contains('Non-small', case=False), 'Non-Small Cell', 'Small Cell')
data['lineage'].value_counts()
columns_to_remove = ['depmap_id', 'cell_line_display_name', 'lineage_1', 'lineage_2', 'lineage_3', 'lineage_5', 'lineage_6', 'lineage_4']
data = data.drop(columns=columns_to_remove)
data.head(2)
y = data.iloc[:, -1]
X = data.iloc[:, :-1]
X.shape