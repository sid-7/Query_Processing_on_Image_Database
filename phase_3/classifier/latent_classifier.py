import numpy as np

from phase_3.config.config_provider import ConfigProvider
from phase_3.mapping.image_factory import ImageFactory
from phase_3.reduction.feature_reduction import PCAReduction
from phase_3.store.class_store import ClassStore
from phase_3.store.feature_store import FeatureStorage, ModelType


class LatentClassifier:

    def __init__(self, _storage_path_train, _storage_path_test, _info_path, _model_type):
        self.storage_path_train = _storage_path_train
        self.storage_path_test = _storage_path_test
        self.info_path = _info_path
        self.model_type = _model_type
        self.feature_storage_train = FeatureStorage(_storage_path_train)
        self.feature_storage_test = FeatureStorage(_storage_path_test)
        self.class_storage = ClassStore(_storage_path_train, _info_path)
        self.reduction = PCAReduction(storage_path=self.storage_path_train,
                                      info_path=self.info_path,
                                      model_type=self.model_type)

    def get_dorsal_image_vectors(self):
        dorsal_image_ids = self.class_storage.get_dorsal_image_ids()
        df = self.feature_storage_train.load_to_df(self.model_type)
        df = df[df.index.isin(list(dorsal_image_ids))]
        return df

    def get_palmer_image_vectors(self):
        palmer_image_ids = self.class_storage.get_palmer_image_ids()
        df = self.feature_storage_train.load_to_df(self.model_type)
        df = df[df.index.isin(list(palmer_image_ids))]
        return df

    def get_highest_dot_product(self, _image_id, _D, _k):
        self.reduction.df = _D
        VT = self.reduction.get_v(_k)
        d = self.feature_storage_test.get_single_data(self.model_type, image_id=_image_id)
        dp = max([np.dot(d, v) for v in VT])
        return dp[0]

    def classify_image(self, _k, _image_id):
        D1 = self.get_dorsal_image_vectors()
        D2 = self.get_palmer_image_vectors()
        dorsal = self.get_highest_dot_product(_image_id, D1, _k)
        palmer = self.get_highest_dot_product(_image_id, D2, _k)
        return "dorsal" if dorsal > palmer else "palmer"


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config.json"
    config_provider = ConfigProvider(config_path)

    storage_path_train = config_provider.get_storage_path_train()
    storage_path_test = config_provider.get_storage_path_test()
    info_path = config_provider.get_info_path()

    latent_classifier = LatentClassifier(_storage_path_train=storage_path_train,
                                         _storage_path_test=storage_path_test,
                                         _info_path=info_path,
                                         _model_type=ModelType.HOG)

    image_factory = ImageFactory(config_provider.get_image_base_path_test())

    image_ids = [image_factory.get_image_id_from_path(ip) for ip in image_factory.get_image_path_list()]
    _dorsal_image_ids = latent_classifier.class_storage.get_dorsal_image_ids()

    correct = 0
    for image_id in image_ids:
        actual = "dorsal" if image_id in _dorsal_image_ids else "palmer"
        pred = latent_classifier.classify_image(_k=10, _image_id=int(image_id))
        if actual == pred:
            correct = correct + 1

    print float(correct) / len(image_ids)
