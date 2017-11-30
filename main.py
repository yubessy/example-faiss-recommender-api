import pandas
import scipy.sparse
from sklearn.decomposition import NMF


def main():
    model = NMF(n_components=2, init='random', random_state=0)
    r = pandas.read_csv("movielens-small/ratings/csv")
    R = scipy.sparse.coo_matrix(r['raring'], (r['userId'], r['itemId']))
    U = model.fit_transform(R)
    I = model.components_


if __name__ == '__main__':
    main()
