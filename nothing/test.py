import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
import mglearn
import warnings
import pandas as pd

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=0)


def f1():
    from sklearn.datasets import fetch_lfw_people

    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    msk = np.zeros(people.target.shape, dtype=np.bool)

    for target in np.unique(people.target):
        msk[np.where(people.target == target)[0][:50]] = 1

    X_people = people.date[msk]
    y_people = people.target[msk]

    X_people /= 255.


def f2():
    import mglearn
    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split

    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    msk = np.zeros(people.target.shape, dtype=np.bool)
    print(image_shape)
    for target in np.unique(people.target):
        msk[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[msk]
    y_people = people.target[msk]

    X_people /= 255.

    X_train, y_train, X_test, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

    print(X_train.shape)
    print(image_shape)
    mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
    plt.show()


def GetXtrain():
    from sklearn.datasets import fetch_lfw_people
    from sklearn.model_selection import train_test_split

    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    msk = np.zeros(people.target.shape, dtype=np.bool)

    for target in np.unique(people.target):
        msk[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[msk]
    y_people = people.target[msk]

    X_people /= 255.

    return train_test_split(X_people, y_people, stratify=y_people, random_state=0)


def f3():
    import mglearn
    from sklearn.decomposition import NMF
    X_people, y_people, im_shape = GetXy()
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

    mglearn.plots.plot_nmf_faces(X_train, X_test, im_shape)
    plt.show()


def GetXy():
    from sklearn.datasets import fetch_lfw_people
    import numpy as np

    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    """
    fix,axes = plt.subplots(2,5,figsize=(15,18),
                            subplot_kw={'xticks':(),
                                        'yticks':()})

    for target, image, ax in zip(people.target,people.images,axes.ravel()):
        ax.imshow(image)
        ax.set_title(people.target_names[target])
    plt.show()
    """
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[mask]
    y_people = people.target[mask]

    X_people = X_people / 255.

    return X_people, y_people, image_shape


def f4():
    from sklearn.decomposition import NMF
    from sklearn.model_selection import train_test_split

    X_people, y_people, im_shape = GetXy()
    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

    nmf = NMF(n_components=15, random_state=0).fit(X_train)
    X_train_nmf = nmf.transform(X_train)
    X_test_nmf = nmf.transform(X_test)

    fix, axes = plt.subplots(3, 5, figsize=(15, 12),
                             subplot_kw={'xticks': (), 'yticks': ()})
    """
    for i, (component,ax) in enumerate(zip(nmf.components_,axes.ravel())):
        ax.imshow(component.reshape(im_shape))
        ax.set_title("{}. component".format(i))
    plt.show()
    """

    comp = 3
    inds = np.argsort(X_train_nmf[:, comp])[::-1]
    fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                             subplot_kw={'xticks': (), 'yticks': ()})

    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(im_shape))

    plt.show()

    comp = 7

    inds = np.argsort(X_train_nmf[:, comp])[::-1]
    fig, axes = plt.subplots(2, 5, figsize=(15, 8),
                             subplot_kw={'xticks': (), 'yticks': ()})

    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(im_shape))

    plt.show()


"""A signal that consists of three different sources"""


def f5():
    from mglearn.datasets import make_signals
    from sklearn.decomposition import NMF

    S = make_signals()
    plt.figure(figsize=(6, 1))
    plt.plot(S, '-')
    plt.xlabel("Time")
    plt.ylabel("Signal")

    A = np.random.RandomState(0).uniform(size=(100, 3))
    X = np.dot(S, A.T)

    nmf = NMF(n_components=3, random_state=42)
    S_ = nmf.fit_transform(X)
    print("Recovered signal shape {}".format(S_.shape))

    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    H = pca.fit_transform(X)

    models = [X, S, S_, H]

    names = ['Observations (first three measurements)',
             'True source',
             'NMF',
             'PCA']

    fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5},
                             subplot_kw={'xticks': (), 'yticks': ()})

    for model, name, ax in zip(models, names, axes):
        ax.set_title(name)
        ax.plot(model[:, :3], '-')
    plt.show()


def f6():
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA

    digits = load_digits()

    pca = PCA(n_components=2)
    pca.fit(digits.data)
    digits_pca = pca.transform(digits.data)

    colors = ['#476A2A', '#7851B8', "#BD3430", '#4A2D4E', "#875525",
              "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]

    plt.figure(figsize=(10, 10))
    plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
    plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())

    for i in range(len(digits.data)):
        plt.text(digits_pca[i, 0], digits_pca[i, 1], str(digits.target[i]),
                 color=colors[digits.target[i]],
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.show()


def f7():
    from sklearn.manifold import TSNE
    from sklearn.datasets import load_digits
    colors = ['#476A2A', '#7851B8', "#BD3430", '#4A2D4E', "#875525",
              "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
    digits = load_digits()

    tsne = TSNE(random_state=42)
    digits_tsne = tsne.fit_transform(digits.data)

    plt.figure(figsize=(10, 10))
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)

    for i in range(len(digits.data)):
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
                 color=colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})

    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.show()


def f8():
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans

    X, y = make_blobs(random_state=1)
    km = KMeans(n_clusters=3)
    km.fit(X)

    print("Cluster membership:\n{}".format(km.labels_))
    print("---------------")

    """
    mglearn.discrete_scatter(X[:,0],X[:,1],km.labels_,markers='o')
    mglearn.discrete_scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],[0,1,2],markers='^',
                             markeredgewidth=2)
    """

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)

    assignments = kmeans.labels_

    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

    km = KMeans(n_clusters=5)
    km.fit(X)
    assignments = km.labels_

    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])
    plt.show()


"""Comparison of PCA NMF and kMeans"""


def f9():
    from sklearn.decomposition import NMF, PCA
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    X_people, y_people, im_shape = GetXy()
    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)
    nmf = NMF(n_components=100, random_state=0)
    nmf.fit(X_train)

    pca = PCA(n_components=100, random_state=0)
    pca.fit(X_train)

    kmeans = KMeans(n_clusters=100, random_state=0)
    kmeans.fit(X_train)

    X_pca = pca.inverse_transform(pca.transform(X_test))
    X_nmf = np.dot(nmf.transform(X_test), nmf.components_)
    X_km = kmeans.cluster_centers_[kmeans.predict(X_test)]

    fig, axes = plt.subplots(3, 5, figsize=(8, 8),
                             subplot_kw={'xticks': (), 'yticks': ()})

    fig.suptitle("Extracted components")

    for ax, comp_km, comp_pca, comp_nmf in zip(axes.T, kmeans.cluster_centers_, pca.components_,
                                               nmf.components_):
        ax[0].imshow(comp_km.reshape(im_shape))
        ax[1].imshow(comp_pca.reshape(im_shape), cmap='viridis')
        ax[2].imshow(comp_nmf.reshape(im_shape))

    axes[0, 0].set_ylabel("kmeans")
    axes[1, 0].set_ylabel("pca")
    axes[2, 0].set_ylabel("nmf")

    fig, axes = plt.subplots(4, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
    fig.suptitle("Reconstruction")

    for ax, orig, rec_km, rec_pca, rec_nmf in zip(
            axes.T, X_test, X_km, X_pca, X_nmf):
        ax[0].imshow(orig.reshape(im_shape))
        ax[1].imshow(rec_km.reshape(im_shape))
        ax[2].imshow(rec_pca.reshape(im_shape))
        ax[3].imshow(rec_nmf.reshape(im_shape))

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("km")
    axes[2, 0].set_ylabel("pca")
    axes[3, 0].set_ylabel("nmf")

    plt.show()


"""Agglomerative clustering"""


def f10():
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import make_blobs

    X, y = make_blobs(random_state=1)
    agg = AgglomerativeClustering(n_clusters=3)
    assignment = agg.fit_predict(X)

    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)

    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.show()


def f11():
    from scipy.cluster.hierarchy import dendrogram, ward
    from sklearn.datasets import make_blobs
    X, y = make_blobs(random_state=0, n_samples=12)

    linkage_array = ward(X)
    dendrogram(linkage_array)

    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [7.25, 7.25], '--', c='k')
    ax.plot(bounds, [4, 4], '--', c='k')

    ax.text(bounds[1], 7.25, "two clusters", va='center', fontdict={'size': 15})
    ax.text(bounds[1], 4, "two clusters", va='center', fontdict={'size': 15})
    plt.xlabel("Two clusters")
    plt.ylabel("three clusters")

    plt.show()


def f12():
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_blobs

    X, y = make_blobs(random_state=0, n_samples=12)

    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X)
    print('Cluster membership:\n{}'.format(clusters))
    mglearn.plots.plot_dbscan()
    plt.show()


def f13():
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    scaler = StandardScaler()
    scaler.fit(X)
    X_transformed = scaler.transform(X)

    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X_transformed)
    plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
    plt.show()


def fg1():
    from sklearn.metrics.cluster import adjusted_mutual_info_score, silhouette_score
    from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    X, y = make_moons(n_samples=500, noise=0.07, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3),
                             subplot_kw={'xticks': (), 'yticks': ()})

    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X))

    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    axes[0].set_title("Random assignment - ARI: {:.2f}".format(
        silhouette_score(X_scaled, random_clusters)))

    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3, s=30)
        ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,
                                               silhouette_score(X_scaled, clusters)))

    plt.show()


"""Comparing all those algorithms on faces dataset"""


def fg2():
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN

    X_people, y_people, img_shape = GetXy()
    pca = PCA(whiten=True, n_components=100, random_state=0)
    pca.fit(X_people)
    X_pca = pca.transform(X_people)

    for eps in [1, 3, 5, 7, 9, 11, 13]:
        print("\neps={}".format(eps))
        dbscan = DBSCAN(min_samples=3, eps=eps)
        labels = dbscan.fit_predict(X_pca)

        print("Unique labels: {}".format(np.unique(labels)))
        print("Number of points per cluster: {}".format(np.bincount(labels + 1)))

    noise = X_people[labels == -1]

    fig, axes = plt.subplots(4, 8, figsize=(12, 4),
                             subplot_kw={'xticks': (), 'yticks': ()})
    for image, ax in zip(noise, axes.ravel()):
        ax.imshow(image.reshape(img_shape), vmin=0, vmax=1)
    plt.show()


def fg3():
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import fetch_lfw_people

    X_people, y_people, img_shape = GetXy()
    pca = PCA(whiten=True, n_components=100, random_state=0)
    pca.fit(X_people)
    X_pca = pca.transform(X_people)
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

    dbscan = DBSCAN(min_samples=3, eps=7)
    labels = dbscan.fit_predict(X_pca)

    for cluster in range(max(labels) + 1):
        mask = labels == cluster
        n_images = np.sum(mask)
        fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4),
                                 subplot_kw={'xticks': (), 'yticks': ()})

        for image, label, ax in zip(X_people[mask], y_people[mask], axes):
            ax.imshow(image.reshape(img_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1])

    plt.show()


def fg4():
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.datasets import fetch_lfw_people

    X_people, y_people, img_shape = GetXy()
    pca = PCA(whiten=True, n_components=100, random_state=0)
    pca.fit(X_people)
    X_pca = pca.transform(X_people)
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

    km = KMeans(n_clusters=10, random_state=0)
    labels = km.fit_predict(X_pca)
    print("Cluster sizes kmeans: {}".format(np.bincount(labels)))

    fig, axes = plt.subplots(2, 5, figsize=(12, 4),
                             subplot_kw={'xticks': (), 'yticks': ()})

    for center, ax in zip(km.cluster_centers_, axes.ravel()):
        ax.imshow(pca.inverse_transform(center).reshape(img_shape), vmin=0, vmax=1)

    mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)
    plt.show()


def fg5():
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import fetch_lfw_people
    from scipy.cluster.hierarchy import ward, dendrogram

    X_people, y_people, img_shape = GetXy()
    pca = PCA(whiten=True, n_components=100, random_state=0)
    pca.fit(X_people)
    X_pca = pca.transform(X_people)
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)

    ag = AgglomerativeClustering(n_clusters=40)
    labels = ag.fit_predict(X_pca)

    print("Number of points per cluster: {}".format(np.bincount(labels)))

    linkage_array = ward(X_pca)

    n_clusters = 40
    for cluster in [10, 13, 19, 22, 36]:
        mask = labels == cluster
        fig, axes = plt.subplots(1, 15, figsize=(15, 8),
                                 subplot_kw={'xticks': (), 'yticks': ()})

        cluster_size = (np.sum(mask))
        axes[0].set_ylabel("#{}: {}".format(cluster, cluster_size))
        for image, label, asdf, ax in zip(X_people[mask], y_people[mask],
                                          labels[mask], axes):
            ax.imshow(image.reshape(img_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1],
                         fontdict={'fontsize': 9})
        for i in range(cluster_size, 15):
            axes[i].set_visible(False)
    plt.show()


def fg6():
    data = pd.read_csv('adult.data', header=None, index_col=False, names=
    ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
     'occupation', 'relationship', 'race', 'gender',
     'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
    data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']]

    display(data.head())
    print("Original features:\n", list(data.columns), '\n')
    data_dummies = pd.get_dummies(data)
    print("Dummies after get_dummies:\n", list(data_dummies.columns), '\n')

    features = data_dummies.ix[:, 'age':'occupation_ Transport-moving']
    X = features.values
    y = data_dummies['income_ >50K'].values
    print('X.sape {} y.shape {}'.format(X.shape, y.shape))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    logreg = LogisticRegression().fit(X_train, y_train)
    print('Test score: {:.2f}'.format(logreg.score(X_test, y_test)))


def fg7():
    demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1],
                            'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
    display(demo_df)

    demo_df['Integer Feature'] = demo_df['Integer Feature'].astype(str)
    dl = pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature'])
    display(dl)


def fg8():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from mglearn.datasets import make_wave
    from sklearn.preprocessing import OneHotEncoder

    X, y = make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

    reg = LinearRegression().fit(X, y)
    plt.plot(line, reg.predict(line), label="linear regression")

    reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
    plt.plot(line, reg.predict(line), label="decision tree")

    plt.plot(X[:, 0], y, 'o', c='r')
    plt.ylabel("regression output")
    plt.xlabel("input features")
    plt.legend(loc='best')

    bins = np.linspace(-3, 3, 11)
    print("bins: {}".format(bins))
    which_bin = np.digitize(X, bins=bins)
    print("\nData points: ", X[:5])
    print("\nBin membership for the data points:\n", which_bin[:5])

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(which_bin)
    X_binned = encoder.transform(which_bin)
    print(X_binned[:5])

    line_binned = encoder.transform(np.digitize(line, bins=bins))
    reg = LinearRegression().fit(X_binned, y)

    plt.show()

    plt.plot(line, reg.predict(line_binned), '--', label="linear regression binned", c='r')

    reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)

    plt.plot(line, reg.predict(line_binned), label="Decision tree binned", alpha=.5)
    plt.plot(X[:, 0], y, 'o', c='k')
    plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
    plt.legend(loc='best')
    plt.ylabel("Regression output")
    plt.xlabel("Input features")
    plt.show()


def fg9():
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from mglearn.datasets import make_wave
    from sklearn.preprocessing import OneHotEncoder

    X, y = make_wave(n_samples=100)
    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)

    bins = np.linspace(-3, 3, 11)
    which_bin = np.digitize(X, bins=bins)

    encoder = OneHotEncoder(sparse=False)
    encoder.fit(which_bin)
    X_binned = encoder.transform(which_bin)
    X_combined = np.hstack([X, X_binned])
    X_product = np.hstack([X_binned, X * X_binned])
    line_binned = encoder.transform(np.digitize(line, bins=bins))
    print(X_product.shape)

    reg = LinearRegression().fit(X_product, y)

    line_combined = np.hstack([line_binned, line * line_binned])
    plt.plot(line, reg.predict(line_combined), label='Linear regression combined')

    for bin in bins:
        plt.plot([bin, bin], [-3, 3], ':', c='k')

    plt.plot(X[:, 0], y, 'o', c='k')
    plt.legend(loc='best')
    plt.ylabel("Regression output")
    plt.xlabel("Input features")
    plt.plot(X[:, 0], y, 'o', c='k')

    plt.show()


def fg10():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from mglearn.datasets import make_wave
    from sklearn.preprocessing import OneHotEncoder

    X, y = make_wave(n_samples=100)
    poly = PolynomialFeatures(degree=10, include_bias=False)
    X_poly = poly.fit_transform(X)
    print("Polynomial feature names: \n{}".format(poly.get_feature_names()))

    reg = LinearRegression().fit(X_poly, y)

    line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
    line_poly = poly.transform(line)
    plt.plot(line, reg.predict(line_poly), label="Polinomial linear regression")
    plt.plot(X[:, 0], y, 'o', c='k')
    plt.ylabel("Regression output")
    plt.xlabel("Input features")
    plt.legend(loc='best')
    plt.show()


def df1():
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import PolynomialFeatures

    boston = load_boston()
    X_train, X_test, y_train, y_test = train_test_split(
        boston.data, boston.target, random_state=0
    )
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)

    poly = PolynomialFeatures(degree=2).fit(X_train_scaled)
    X_train_poly = poly.transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    print(X_train_poly.shape)
    print("Features after poly:\n{}".format(poly.get_feature_names()))

    from sklearn.linear_model import Ridge
    ridge = Ridge().fit(X_train_scaled, y_train)
    print("Score without interactions: {:.3f}".format(ridge.score(X_test_scaled, y_test)))

    ridge = Ridge().fit(X_train_poly, y_train)
    print("score with interactions: {:.3f}".format(ridge.score(X_test_poly, y_test)))

    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(n_estimators=100).fit(X_train_scaled, y_train)
    print("Score without interactions: {:.3f}".format(rf.score(X_test_scaled, y_test)))

    rf = RandomForestRegressor(n_estimators=100).fit(X_train_poly, y_train)
    print("Score with interactions: {:.3f}".format(rf.score(X_test_poly, y_test)))


def df2():
    rnd = np.random.RandomState(0)
    X_org = rnd.normal(size=(1000, 3))
    w = rnd.normal(size=3)

    X = rnd.poisson(10 * np.exp(X_org))
    y = np.dot(X_org, w)

    print("Number of feature appearences:\n{}".format(np.bincount(X[:, 0])))
    bins = np.bincount(X[:, 0])
    plt.bar(range(len(bins)), bins, color='k')
    plt.ylabel("Number of appearances")
    plt.xlabel("Value")
    plt.show()

    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    score = Ridge().fit(X_train, y_train).score(X_test, y_test)
    print("Score: {:.3f}".format(score))

    X_train_log = np.log(X_train + 1)
    X_test_log = np.log(X_test + 1)

    plt.hist(np.log(X_train_log[:, 0] + 1), bins=25, color='gray')
    plt.show()

    score = Ridge().fit(X_train_log, y_train).score(X_test_log, y_test)
    print("Score: {:.3f}".format(score))


def df3():
    from sklearn.datasets import load_breast_cancer
    from sklearn.feature_selection import SelectPercentile
    from sklearn.model_selection import train_test_split

    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len((cancer.data)), 50))
    X_w_noise = np.hstack([cancer.data, noise])

    X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

    select = SelectPercentile(percentile=50).fit(X_train, y_train)
    X_train_selected = select.transform(X_train)

    print("X_train.shape: {}".format(X_train.shape))
    print("X_train_selected.shape: {}".format(X_train_selected.shape))

    mask = select.get_support()
    print(mask)

    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("Sample index")
    plt.show()

    from sklearn.linear_model import LogisticRegression

    X_test_selected = select.transform(X_test)

    lr = LogisticRegression().fit(X_train, y_train)
    print("Score without selection: {:.3f}".format(lr.score(X_test, y_test)))

    lr = LogisticRegression().fit(X_train_selected, y_train)
    print("Score with selection: {:.3f}".format(lr.score(X_test_selected, y_test)))

class Lol:
    def __init__(self):
        pass


def df4():
    from sklearn.feature_selection import SelectFromModel
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len((cancer.data)), 50))
    X_w_noise = np.hstack([cancer.data, noise])

    X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)
    select = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42),
                             threshold='median')

    select.fit(X_train, y_train)
    X_train_l1 = select.transform(X_train)

    print('X_train.shape: {}'.format(X_train.shape))
    print("X_train_l1.shape: {}".format(X_train_l1.shape))

    mask = select.get_support()

    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("Sample index")

    X_test_l1 = select.transform(X_test)
    print("Score: {:.3f}".format(LogisticRegression().fit(X_train_l1, y_train).score(X_test_l1, y_test)))

    plt.show()


def df5():
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    import time
    from sklearn.linear_model import LogisticRegression

    select = RFE(RandomForestClassifier(n_estimators=100, random_state=42),
                 n_features_to_select=40)
    cancer = load_breast_cancer()
    rng = np.random.RandomState(42)
    noise = rng.normal(size=(len((cancer.data)), 50))
    X_w_noise = np.hstack([cancer.data, noise])
    X_train, X_test, y_train, y_test = train_test_split(X_w_noise, cancer.target, random_state=0, test_size=.5)

    start_time = time.time()
    select.fit(X_train, y_train)
    print("Estimated execution time: {} seconds".format((time.time() - start_time)))

    X_train_rfe = select.transform(X_train)
    X_test_rfe = select.transform(X_test)

    score = LogisticRegression().fit(X_train_rfe, y_train).score(X_test_rfe, y_test)
    print("Score: {:.3f}".format(score))

    mask = select.get_support()
    plt.matshow(mask.reshape(1, -1), cmap='gray_r')
    plt.xlabel("Sample index")
    plt.show()


def df6():
    from sklearn.ensemble import RandomForestRegressor
    import time
    import array

    citibike = mglearn.datasets.load_citibike()
    print("Citibike data: \n{}".format(citibike.head()))

    plt.figure(figsize=(10, 3))
    xticks = pd.date_range(start=citibike.index.min(), end=citibike.index.max(), freq='D')

    plt.xticks(xticks, xticks.strftime("%a %m-%d"), rotation=90, ha='left')
    plt.plot(citibike, linewidth=1)
    plt.xlabel("date")
    plt.ylabel('Rentals')
    # plt.show()

    y = citibike.values
    X = array(citibike.index.strftime("%s").astype("int")).reshape(-1, 1)

    n_train = 184

    regressor = RandomForestRegressor(n_estimators=100, random_state=0)

    X_train, X_test = X[:n_train], X[n_train:]

    y_train, y_test = y[:n_train], y[n_train:]
    regressor.fit(X_train, y_train)
    print('Test-set R^2: {:.2f}'.format(regressor.score(X_test, y_test)))

    y_pred = regressor.predict(X_test)
    y_pred_train = regressor.predict(X_train)

    plt.figure(figsize=(10, 3))

    plt.xticks(range(0, len(X), 8), xticks.strftime("%a %m-%d"), rotation=90, ha='left')
    plt.plot(range(n_train), y_train, label='train')
    plt.plot(range(n_train), len(y_test) + n_train, y_test, '-', label='test')
    plt.plot(range(n_train), y_pred_train, '--', label='prediction train')
    plt.plot(range(n_train), len(y_test) + n_train, y_pred, '--', label='prediction test')
    plt.legend(loc='best')
    plt.xlabel("Date")
    plt.ylabel("Rentals")
    plt.show()


df6()
