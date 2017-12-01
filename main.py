import faiss
import numpy
from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF
from flask import Flask, jsonify

RANDOM_STATE = 0
N_FACTOR = 10
N_RESULT = 10

# データロード
ratings = numpy.loadtxt(
    "/movielens-small/ratings.csv",
    delimiter=",",
    skiprows=1,
    usecols=(0, 1, 2),
    dtype=[('userId', 'i8'), ('movieId', 'i8'), ('rating', 'f8')],
)
users = sorted(numpy.unique(ratings['userId']))
user_id2i = {id: i for i, id in enumerate(users)}
movies = sorted(numpy.unique(ratings['movieId']))
movie_id2i = {id: i for i, id in enumerate(movies)}
movie_i2id = {i: id for i, id in enumerate(movies)}
R = coo_matrix(
    (ratings['rating'], (ratings['userId'].map(user_id2i.get),
                         ratings['movieId'].map(movie_id2i.get)))
)

# モデル学習
model = NMF(n_components=N_FACTOR, init='random', random_state=RANDOM_STATE)
U = model.fit_transform(R)
I = model.components_.T

# インデックス作成
movie_index = faiss.IndexFlatIP(N_FACTOR)

# Web API
app = Flask(__name__)


@app.route("/users/<int:user_id>")
def users(user_id):
    user_i = user_id2i[user_id]
    user_vec = U[user_i]
    scores, indices = movie_index.search(numpy.array([user_vec]), N_RESULT)
    item_scores = zip(indices[0], scores[0])
    return jsonify(
        items=[{"id": movie_i2id[i], "score": s} for i, s in item_scores],
    )
