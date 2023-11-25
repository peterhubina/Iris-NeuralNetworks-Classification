from sklearn import datasets
import torch
import pandas as pd

# Load the Iris dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Create a DataFrame with feature data
iris_df = pd.DataFrame(X, columns=iris.feature_names)

# Add a column for the target (species)
iris_df['species'] = pd.Categorical.from_codes(y, iris.target_names)

# Display the first few rows of the DataFrame
print(iris_df.head())