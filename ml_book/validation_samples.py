from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()


def f1():
    from sklearn.model_selection import GroupKFold
    from sklearn.datasets import make_blobs
    from sklearn.model_selection import cross_val_score

    X, y = make_blobs(n_samples=12, random_state=0)

    groups = [0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 3]
    scores = cross_val_score(logreg, X, y, groups, cv=GroupKFold(n_splits=3))
    print("Cross validation scores:\n{}".format(scores))


def f2():




f2()
