# faiss-recommender-api-example

Example app using [facebookresearch/faiss](https://github.com/facebookresearch/faiss) as Web API for NMF based recommender system.

Original article: https://qiita.com/yubessy/private/bae2c6c4d1bee0d8fc0b (in Japanese)

## Usage

```
$ git clone https://github.com/yubessy/faiss-recommender-api-example.git
$ cd faiss-recommender-api-example
$ docker build -t faiss-recommender-api-example .  # takes quite a long time...
$ docker run -p 5000:5000 faiss-recommender-api-example
```

## About dataset

Files under `movielens-small` were downloaded from https://grouplens.org/datasets/movielens/
