from sklearn import datasets
import pandas as pd

def create_iris():
    iris = datasets.load_iris()
    X_iris, y_iris = iris.data, iris.target
    y_iris = [iris.target_names[i] for i in iris.target]

    z0 = pd.DataFrame(X_iris, columns=iris.feature_names)
    z1 = pd.DataFrame(y_iris, columns=["Flower species"])
    z2 = pd.concat([z0, z1], axis=1)
    z2.to_csv("iris_dataset.csv")
