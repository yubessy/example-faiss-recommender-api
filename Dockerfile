FROM nvidia/cuda:8.0-devel-ubuntu16.04

# from: https://github.com/facebookresearch/faiss/blob/master/Dockerfile
# begin
RUN apt-get update -y
RUN apt-get install -y libopenblas-dev python-numpy python-dev swig git python-pip wget

RUN pip install matplotlib

ENV FAISS_REPO_COMMIT e652a66
RUN git clone --depth 1 https://github.com/facebookresearch/faiss.git /opt/faiss && \
    cd /opt/faiss && \
    git checkout $FAISS_REPO_COMMIT

WORKDIR /opt/faiss

ENV BLASLDFLAGS /usr/lib/libopenblas.so.0

RUN mv example_makefiles/makefile.inc.Linux ./makefile.inc

RUN make tests/test_blas -j $(nproc) && \
    make -j $(nproc) && \
    make tests/demo_sift1M -j $(nproc)

RUN make py

RUN cd gpu && \
    make -j $(nproc) && \
    make test/demo_ivfpq_indexing_gpu && \
    make py
# end

RUN pip install --upgrade pip && \
    pip install numpy scipy scikit-learn flask

COPY movielens-small movielens-small
COPY main.py main.py

CMD ["python", "main.py"]
