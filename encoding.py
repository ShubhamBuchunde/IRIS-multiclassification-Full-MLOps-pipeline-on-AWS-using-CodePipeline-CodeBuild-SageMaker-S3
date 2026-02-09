import pandas as pd
from sklearn.datasets import dump_svmlight_file

# Load your dataset
df = pd.read_csv('iris.csv')

# Encode the target variable
df['target_encoded'] = df['target'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
})

# Features and labels
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = df['target_encoded'].values

# Convert to LIBSVM format
dump_svmlight_file(X, y, 'iris.libsvm', zero_based=True)