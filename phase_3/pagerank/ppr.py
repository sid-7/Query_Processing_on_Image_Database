import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix

from phase_3.config.config_provider import ConfigProvider
from phase_3.pagerank.create_similarity_matrix import create_similarity
from phase_3.store.feature_store import FeatureStorage, ModelType


class PPR:

    def __init__(self):
        store_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/sim.pkl"
        self.input_csv_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/input_df.csv"
        self.sim_df = pd.read_pickle(store_path)
        self.normalized = False
        self.A = None
        self.base = None
        self.m = 0
        self.n = 0
        self.node_ids = None
        self.nAT = None

    @staticmethod
    def iterate(A, q):
        x = q
        old_x = q
        residuals = np.zeros(100)
        for i in range(100):
            x = 0.85 * (A.dot(old_x))
            S = np.sum(x)
            x = x + (1 - S) * q
            residuals[i] = norm(x - old_x, 1)
            if residuals[i] <= 1e-9:
                break
            old_x = x
        return x

    @staticmethod
    def __row_normalize(A):
        m, n = A.shape
        d = A.sum(axis=1)
        d = np.asarray(d).flatten()
        d = np.maximum(d, np.ones(n))
        invD = spdiags((1.0 / d), 0, m, n)
        nA = invD.dot(A)
        return nA

    @staticmethod
    def __read_graph(input_path):
        X = np.loadtxt(input_path, dtype=float, comments='#')
        base = np.amin(X[:, 0:2])
        X[:, 0:2] = X[:, 0:2] - base
        n = int(np.amax(X[:, 0:2])) + 1
        A = csr_matrix((X[:, 2], (X[:, 0], X[:, 1])), shape=(n, n))
        return A, base.astype(int)

    def read_graph(self):
        self.A, self.base = self.__read_graph(self.input_csv_path)
        self.m, self.n = self.A.shape
        self.node_ids = np.arange(0, self.n) + self.base
        self.normalize()

    def normalize(self):
        if self.normalized is False:
            nA = self.__row_normalize(self.A)
            self.nAT = nA.T
            self.normalized = True

    def get_k_directed_nodes(self, k):
        result = []
        for i in self.sim_df.index:
            row = self.sim_df.sort_values(by=[i], ascending=[0])
            result.append([str(a) + "," + str(b) for a, b in zip(list(row.index), list(row[i]))])

        sorted_df = pd.DataFrame(data=result, index=self.sim_df.index, columns=self.sim_df.index)

        input_df = pd.DataFrame(columns=["image_id", "other_image_id", "similarity"])

        i = 0
        for row in sorted_df.to_numpy():
            image_id = row[0].split(",")[0]
            for col in row[1:(k + 1)]:
                input_df.loc[i] = [image_id, col.split(",")[0], col.split(",")[1]]
                i = i + 1

        input_df.to_csv(self.input_csv_path, sep="\t", header=False, index=False)

    def compute(self, _seeds, _K, _k):
        self.get_k_directed_nodes(_k)
        self.read_graph()
        self.normalize()

        _seeds = [seed - self.base for seed in _seeds]
        q = np.zeros(self.n)
        q[_seeds] = 1.0 / len(_seeds)
        result = self.iterate(self.nAT, q)
        result_json = []

        for node_id, score in zip(self.node_ids, result):
            result_json.append({"other_image_id": node_id, "score": score})

        result_json.sort(key=lambda a: a["score"], reverse=True)

        return result_json[:_K]


if __name__ == "__main__":
    config_provider = ConfigProvider("/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3"
                                     "/config_task_3.json")
    feature_storage = FeatureStorage(config_provider.get_storage_path_train())
    df = feature_storage.load_to_df(ModelType.LBP)
    create_similarity(df)

    ppr = PPR()
    seeds = [8333, 6183, 74]
    K = 10
    k = 5

    similar = ppr.compute(seeds, K, k)

    print similar
