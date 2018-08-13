import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import KNN


def getDataset():
    df = pd.read_csv('Base/adult.csv')
    return np.split(df, [len(df.columns) - 1], axis=1)


def getFolds(dataset):
    kf = KFold(n_splits=5, shuffle=True)
    return kf.split(dataset)


k_value = 3

for i in range(0, 3):
    X, Y = getDataset()
    media_result = []
    for i in range(0, 10):
        kfolds = getFolds(X)
        fold_result = []
        for fold in kfolds:
            X_train, X_test = X.iloc[fold[0]], X.iloc[fold[1]]
            Y_train, Y_test = Y.iloc[fold[0]], Y.iloc[fold[1]]
            classifier = KNN.KNN()
            classifier.train(X_train, Y_train)
            fold_result.append(classifier.predict(X_test, Y_test, k=k_value, neighbor_mode=0, vote_mode=0))
        print("Execucao:", i, k_value)
        print(np.sum(fold_result) / 5)
        media_result.append(np.sum(fold_result) / 5)
    if k_value == 3:
        k_value = 5
    else:
        k_value = 7
    print("Media folds")
    print(np.sum(media_result) / 10)
