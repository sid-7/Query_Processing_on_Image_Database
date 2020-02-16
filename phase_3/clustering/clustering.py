import numpy as np
import scipy
from enum import Enum

from phase_3.config.config_provider import ConfigProvider
from phase_3.mapping.image_factory import ImageFactory
from phase_3.reduction.feature_reduction import NMFReduction, PCAReduction
from phase_3.store.class_store import ClassStore
from phase_3.store.feature_store import ModelType, FeatureStorage

np.seterr(divide='ignore', invalid='ignore')


class Orientation(Enum):
    DORSAL = "dorsal"
    PALMER = "palmer"


class Clustering:

    def __init__(self, _storage_path_train, _storage_path_test, _info_path, _image_base_path_train, _model_type):
        self.image_base_path_train = _image_base_path_train
        self.class_storage = ClassStore(_storage_path_train, _info_path)
        self.feature_storage_train = FeatureStorage(_storage_path_train)
        self.feature_storage_test = FeatureStorage(_storage_path_test)
        self.model_type = _model_type
        self.info_path = _info_path
        self.storage_path_test = _storage_path_test

    def get_image_vectors_train(self, orientation):
        image_ids = self.class_storage.get_dorsal_image_ids() \
            if orientation is Orientation.DORSAL \
            else self.class_storage.get_palmer_image_ids()
        df = self.feature_storage_train.load_to_df(self.model_type)
        df = df[df.index.isin(list(image_ids))]
        return df

    def get_image_vectors_test(self):
        df = self.feature_storage_test.load_to_df(self.model_type)
        return df

    def get_cluster_memberships(self, c, orientation):
        return {}


class NMFClustering(Clustering):

    def __init__(self, _storage_path_train, _info_path, _image_base_path_train, _model_type):
        self.reduction = NMFReduction(_storage_path_train, _model_type, _info_path)
        Clustering.__init__(self, _storage_path_train, _info_path, _image_base_path_train, _model_type)

    def get_cluster_memberships(self, c, orientation):
        self.reduction.df = self.get_image_vectors_train(orientation).T
        VT = self.reduction.get_v(c)
        cluster_memberships = [np.argmin(v) for v in VT.T]
        graph = {}
        for image_id, cluster_membership in zip(self.reduction.df.T.index, cluster_memberships):
            if cluster_membership in graph:
                graph[cluster_membership].append(image_id)
            else:
                graph[cluster_membership] = [image_id]
        return graph


class KMeansClustering(Clustering):

    def __init__(self, _storage_path_train, _storage_path_test, _info_path, _image_base_path_train, _model_type):
        self.reduction = PCAReduction(_storage_path_train, _model_type, _info_path)
        self.cluster_centers = []
        Clustering.__init__(self,
                            _storage_path_train, _storage_path_test, _info_path, _image_base_path_train, _model_type)

    # Function to fit k_means clusters and find J (error)
    def get_cluster_memberships(self, c, orientation):
        self.reduction.df = self.get_image_vectors_train(orientation)
        _data = self.reduction.get_u(20)
        _cluster_centers = self.initialize_cluster_centers(_data, c)
        cluster_memberships = [self.cluster_membership(x, _cluster_centers) for x in _data]

        _updated_cluster_centers = []

        bool_array = []

        while False not in [item for sublist in bool_array for item in sublist]:
            bool_array = [[i == j for i, j in zip(x, y)] for x, y in zip(_updated_cluster_centers, _cluster_centers)]
            _updated_cluster_centers = _cluster_centers
            _cluster_centers = self.updated_cluster_centers(cluster_memberships, c, _data)
            cluster_memberships = [self.cluster_membership(x, _cluster_centers) for x in _data]

        cluster_memberships = [m.index(1) for m in cluster_memberships]
        graph = {}

        self.cluster_centers = _cluster_centers

        for image_id, cluster_membership in zip(self.reduction.df.index, cluster_memberships):
            if cluster_membership in graph:
                graph[cluster_membership].append(image_id)
            else:
                graph[cluster_membership] = [image_id]

        return graph

    # Function to find membership array using data 'x' w.r.t cluster_centers
    def cluster_membership(self, x, cluster_centers):
        distances = [self.d(c, x) for c in cluster_centers]
        min_distance = min(distances)
        _cluster_membership = [1 if distance == min_distance else 0 for distance in distances]
        return _cluster_membership

    # Function to find distance between data 'x' and cluster center 'c'
    @staticmethod
    def d(c, x):
        _d = np.linalg.norm(x - c) ** 2
        return _d

    # Initialize cluster centers using k-means++ algorithm
    @staticmethod
    def initialize_cluster_centers(_x, _k):
        _c = [_x[0]]
        for k in range(1, _k):
            d2 = scipy.array([min([scipy.inner(c - x, c - x) for c in _c]) for x in _x])
            ps = d2 / d2.sum()
            c_ps = ps.cumsum()
            r = scipy.rand()
            i = 0
            for j, p in enumerate(c_ps):
                if r < p:
                    i = j
                    break
            _c.append(_x[i])
        return _c

    # Find error using object_function
    def object_function(self, _cluster_centers, _memberships, _data):
        x_distances = [self.d(_cluster_centers[m.index(1)], x) for x, m in zip(_data, _memberships)]
        return sum(x_distances)

    # Find updated cluster centers using new memberships
    @staticmethod
    def updated_cluster_centers(_memberships, _k, _data):
        _memberships_counts = [[m for m in _memberships if m[k] == 1].__len__() for k in range(0, _k)]

        _dimensions = _data.shape[1]

        _new_cluster_centers = [list(np.zeros(_dimensions)) for j in range(0, _k)]

        for x, m in zip(_data, _memberships):
            _new_cluster_centers[m.index(1)] = _new_cluster_centers[m.index(1)] + x

        _new_cluster_centers = [np.true_divide(_c_c, _m_c) for _c_c, _m_c in
                                zip(_new_cluster_centers, _memberships_counts)]

        return _new_cluster_centers

    def get_predict_class(self, _cc_dorsal, _cc_palmer):
        self.reduction.df = self.get_image_vectors_test()
        X = self.reduction.get_u(20)
        distances_dorsal = [sum([self.d(cc, x) for cc in _cc_dorsal]) for x in X]
        distances_palmer = [sum([self.d(cc, x) for cc in _cc_palmer]) for x in X]
        return ["dorsal" if d < p else "palmer" for d, p in zip(distances_dorsal, distances_palmer)]


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_2.json"
    config_provider = ConfigProvider(config_path)

    storage_path_train = config_provider.get_storage_path_train()
    storage_path_test = config_provider.get_storage_path_test()
    image_base_path_train = config_provider.get_image_base_path_train()
    info_path = config_provider.get_info_path()

    '''
    nmf_clustering = NMFClustering(_storage_path_train=storage_path_train,
                                   _info_path=info_path,
                                   _image_base_path_train=image_base_path_train,
                                   _model_type=ModelType.HOG)

    nmf_clustering.plot_cluster_sample(8, Orientation.DORSAL, 8)
    '''

    kmeans_clustering = KMeansClustering(_storage_path_train=storage_path_train,
                                         _storage_path_test=storage_path_test,
                                         _info_path=info_path,
                                         _image_base_path_train=image_base_path_train,
                                         _model_type=ModelType.HOG)

    kmeans_clustering.get_cluster_memberships(c=5, orientation=Orientation.DORSAL)
    cc_dorsal = kmeans_clustering.cluster_centers
    kmeans_clustering.get_cluster_memberships(c=5, orientation=Orientation.PALMER)
    cc_palmer = kmeans_clustering.cluster_centers

    image_factory = ImageFactory(config_provider.get_image_base_path_test())
    test_image_ids = [image_factory.get_image_id_from_path(ip) for ip in image_factory.get_image_path_list()]
    _dorsal_image_ids = kmeans_clustering.class_storage.get_dorsal_image_ids()

    Y_pred = kmeans_clustering.get_predict_class(cc_dorsal, cc_palmer)
    correct = 0
    for i, image_id in enumerate(test_image_ids):
        actual = "dorsal" if image_id in _dorsal_image_ids else "palmer"
        pred = Y_pred[i]
        if actual == pred:
            correct = correct + 1

    print float(correct) / len(test_image_ids)
