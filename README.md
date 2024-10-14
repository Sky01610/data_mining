# Project Title

This project is a machine learning application that uses various algorithms to classify lung cancer into two types: Non-Small Cell and Small Cell. The project is implemented in Python and uses libraries such as pandas, numpy, sklearn, matplotlib, and tensorflow.

## Files in the Project

1. `load_and_process_data.py`: This file loads the dataset from a CSV file, preprocesses the data, and splits it into features (X) and target (y).

2. `Classification.py`: This file imports the processed data from `load_and_process_data.py`, checks for any NaN values, and splits the data into training and testing sets.

3. `Clustering%20by%20KMeans.py`: This file performs KMeans clustering on the data and visualizes the clusters using PCA.

4. `Logistic%20Regression.py`: This file implements a Logistic Regression model for the classification task. It uses GridSearchCV for hyperparameter tuning and computes the model's performance metrics.

5. `SVC.py`: This file implements a Support Vector Classifier (SVC) for the classification task. It uses GridSearchCV for hyperparameter tuning and computes the model's performance metrics.

6. `Analysis%20of%20Predictive%20Features.py`: This file analyzes the importance of the features used in the classification task.

7. `RandomForest.py`: This file implements a Random Forest Classifier for the classification task. It uses GridSearchCV for hyperparameter tuning and computes the model's performance metrics.

8. `Network.py`: This file implements a Neural Network model using TensorFlow and Keras for the classification task. It uses early stopping to prevent overfitting and computes the model's performance metrics.

## How to Run the Project

1. Ensure that you have Python 3.7 or later installed on your machine.

2. Install the required libraries by running the following command in your terminal:
    ```
    pip install pandas numpy sklearn matplotlib tensorflow keras
    ```

3. Clone the project to your local machine.

4. Run each Python file in the order they are listed above.

## Results

The performance of each model is evaluated using accuracy, confusion matrix, and ROC curves. The results are printed to the console and also visualized using matplotlib.

## Future Work

Future work could include trying out more machine learning algorithms, tuning the hyperparameters further, or using a larger dataset for training the models.# data_mining
