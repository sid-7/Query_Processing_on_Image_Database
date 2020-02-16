import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
import sys


sys.path.append("..")
from compare.similarity_function import EuclideanSimilarityFunction
from config.config_provider import ConfigProvider
from store.class_store import ClassStore
from store.feature_store import FeatureStorage
from store.feature_store import ModelType


class FeatureReduction:

    # 0 = gender, 1 = dorsal, 2 = accessories, 3 = left
    # Works for both whole DF which has all the data and narrow DF which has data based on info
    def get_df(self, model_type, info):
        query = {
            "male": {"key": 0, "value": 1},
            "female": {"key": 0, "value": 0},
            "dorsal": {"key": 1, "value": 1},
            "palmer": {"key": 1, "value": 0},
            "with-accessories": {"key": 2, "value": 1},
            "without-accessories": {"key": 2, "value": 0},
            "left-hand": {"key": 3, "value": 1},
            "right-hand": {"key": 3, "value": 0},
        }
        if info is None:
            return self.feature_storage.load_to_df(model_type), None
        elif info not in query:
            raise ValueError('info not proper')
        else:
            class_df = self.class_storage.load_to_df()
            class_df = class_df[class_df[query[info]["key"]] == query[info]["value"]]
            feature_df = self.feature_storage.load_to_df(model_type)
            feature_df = feature_df[feature_df.index.isin(class_df.index)]
            class_df = class_df[class_df.index.isin(feature_df.index)]
            return feature_df, class_df[query[info]["key"]].to_numpy()

    def __init__(self, storage_path, model_type, info_path, info=None):
        self.switcher_pkl = {
            ModelType.LBP: "lbp.pkl",
            ModelType.COLOR_MOMENTS: "color_moments.pkl",
            ModelType.HOG: "hog.pkl",
            ModelType.SIFT: "sift.pkl",
        }
        self.feature_storage = FeatureStorage(storage_path)
        self.class_storage = ClassStore(_info_path=info_path, _storage_path=storage_path)
        self.df, self.Y = self.get_df(model_type, info)

    # Task 1 - Data Semantics
    @staticmethod
    def get_latent_data_term_weight_pairs(data_semantics):
        k = len(data_semantics[0]) - 1 if len(data_semantics) > 0 else 0
        latent_scores = []
        for _k in range(0, k):
            data_semantics.sort(key=lambda j: j['latent_semantic_' + str(_k)], reverse=True)
            image_weights = [(r["image_id"], r['latent_semantic_' + str(_k)]) for r in data_semantics]
            latent_scores.append(image_weights)

        return latent_scores

    # Override differently for every reduction impl
    def get_u(self, k):
        return []

    # Override differently for every reduction impl
    def get_v(self, k):
        return []

    def get_data_latent_semantics(self, k):
        # Each json in this is of the form: {"image_id": 000, "latent_semantic_1": ...}
        _data_latent_semantics = []

        for i, u in zip(self.df.index, self.get_u(k)):
            j = {"image_id": i}
            for x, l in enumerate(u):
                j["latent_semantic_" + str(x)] = l
            _data_latent_semantics.append(j)

        return _data_latent_semantics

    # Task 1 - Feature Semantics
    def get_feature_latent_semantics(self, k):
        result = []
        D = self.df.to_numpy()
        latent_semantix_index = 0
        # For each latent semantic
        for v in self.get_v(k):
            latent_semantic_scores = []
            # Get image with highest dot product with that latent semantic
            for i, d in zip(self.df.index, D):
                latent_semantic_scores.append({"image_id": i, "dot": np.dot(v, d)})
            latent_semantic_scores.sort(key=lambda j: j['dot'], reverse=True)
            result.append({"k": latent_semantix_index,
                           "image_id": latent_semantic_scores[0]["image_id"],
                           "dot": latent_semantic_scores[0]["dot"]})
            latent_semantix_index = latent_semantix_index + 1

        # The return here is form of [{'image_id': 000, 'k': 0, 'dot': Highest} ...]
        return result

    # Task 2 - Get m similar images
    def get_k_similar_from_data_semantics(self, k, m, image_id):
        u_df = pd.DataFrame(data=self.get_u(k), index=self.df.index)
        similarity_function = EuclideanSimilarityFunction(u_df)
        distances_json = similarity_function.get_image_distances(image_id)[:m]
        return distances_json

    def get_x_y(self, k):
        return pd.DataFrame(data=self.get_u(k), index=self.df.index), self.Y


class SVDReduction(FeatureReduction):

    def get_u(self, k):
        U, Sigma, VT = randomized_svd(self.df.to_numpy(), n_components=k, n_iter=5, random_state=None)
        return U

    def get_v(self, k):
        U, Sigma, VT = randomized_svd(self.df.to_numpy(), n_components=k, n_iter=5, random_state=None)
        return VT


class PCAReduction(FeatureReduction):

    def get_u(self, k):
        pca = PCA(n_components=k)
        U, Sigma, VT = pca._fit(self.df.to_numpy())
        U = np.array([u[:k] for u in U])
        return U

    def get_v(self, k):
        pca = PCA(n_components=k)
        U, Sigma, VT = pca._fit(self.df.to_numpy())
        return VT


class NMFReduction(FeatureReduction):

    def get_u(self, k):
        model = NMF(n_components=k, init='random', random_state=0)
        U = model.fit_transform(self.df)
        return U

    def get_v(self, k):
        model = NMF(n_components=k, init='random', random_state=0)
        model.fit_transform(self.df)
        VT = model.components_
        return VT


class LDAReduction(FeatureReduction):

    def get_u(self, k):
        lda = LatentDirichletAllocation(n_components=k)
        diff = abs(self.df.values.min())
        self.df += diff
        lda_f = lda.fit(self.df)
        U = lda_f.transform(self.df)
        return U

    def get_v(self, k):
        lda = LatentDirichletAllocation(n_components=k)
        lda_f = lda.fit(self.df)
        VT = lda_f.components_
        return VT


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_2/config.json"
    config_provider = ConfigProvider(config_path)

    # region SVD, LBP
    reduction = SVDReduction(storage_path=config_provider.get_storage_path(),
                             info_path=config_provider.get_info_path(),
                             model_type=ModelType.LBP)

    # Task 1 Test
    data_latent_semantics = reduction.get_data_latent_semantics(k=10)
    print data_latent_semantics
    print reduction.get_latent_data_term_weight_pairs(data_latent_semantics)
    print reduction.get_feature_latent_semantics(10)

    # Task 2 Test
    print reduction.get_k_similar_from_data_semantics(k=10, m=3, image_id=2107)
    # endregion

    # region PCA, LBP
    reduction = PCAReduction(storage_path=config_provider.get_storage_path(),
                             info_path=config_provider.get_info_path(),
                             model_type=ModelType.LBP)

    # Task 1 Test
    data_latent_semantics = reduction.get_data_latent_semantics(k=10)
    # print data_latent_semantics
    # print reduction.get_latent_data_term_weight_pairs(data_latent_semantics)
    # print reduction.get_feature_latent_semantics(10)

    # Task 2 Test
    # print reduction.get_k_similar_from_data_semantics(k=10, m=3, image_id=2108)
    # endregion

    # region NMF, LBP
    reduction = NMFReduction(storage_path=config_provider.get_storage_path(),
                             info_path=config_provider.get_info_path(),
                             model_type=ModelType.LBP)

    # Task 1 Test
    data_latent_semantics = reduction.get_data_latent_semantics(k=10)
    # print data_latent_semantics
    # print reduction.get_latent_data_term_weight_pairs(data_latent_semantics)
    # print reduction.get_feature_latent_semantics(10)

    # Task 2 Test
    # print reduction.get_k_similar_from_data_semantics(k=10, m=3, image_id=2108)
    # endregion

    # region LDA, LBP
    reduction = LDAReduction(storage_path=config_provider.get_storage_path(),
                             info_path=config_provider.get_info_path(),
                             model_type=ModelType.LBP)

    # Task 1 Test
    data_latent_semantics = reduction.get_data_latent_semantics(k=10)
    # print data_latent_semantics
    # print reduction.get_latent_data_term_weight_pairs(data_latent_semantics)
    # print reduction.get_feature_latent_semantics(10)

    # Task 2 Test
    # print reduction.get_k_similar_from_data_semantics(k=10, m=3, image_id=2108)
    # endregion
