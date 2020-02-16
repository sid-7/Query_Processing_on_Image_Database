import json

import pandas as pd
from phase_2.mapping.image_factory import ImageFactory
from phase_2.config.config_provider import ConfigProvider
import itertools


# 0 = gender, 1 = dorsal, 2 = accessories, 3 = left
class ClassStore:

    def __init__(self, _storage_path, _info_path):
        self.df = pd.read_csv(_info_path, sep=",")
        self.image_factory = ImageFactory("")
        self.storage_path = _storage_path

    def get_class_vectors(self):
        class_vectors = {}
        for image_name in self.df["imageName"]:
            image_id = self.image_factory.get_image_id_from_path(image_name)
            idf = self.df[self.df["imageName"] == image_name]
            gender = 1 if idf["gender"].values[0] == "male" else 0
            dorsal = 1 if idf["aspectOfHand"].values[0].split(" ")[0] == "dorsal" else 0
            accessories = idf["accessories"].values[0]
            left = 1 if idf["aspectOfHand"].values[0].split(" ")[1] == "left" else 0
            class_vectors[image_id] = [gender, dorsal, accessories, left]
        return class_vectors

    def store_class_set(self, class_vectors):
        db_file = self.storage_path + "hand_classes.json"
        with open(db_file, 'w') as f:
            json.dump(class_vectors, f)
        self.store_to_pkl()

    def load_to_df(self):
        db_file = self.storage_path + "hand_classes.pkl"
        _df = pd.read_pickle(db_file)
        return _df

    def store_to_pkl(self):
        db_file = self.storage_path + "hand_classes.json"
        _df = pd.read_json(db_file)
        _df = _df.T
        _df.to_pickle(self.storage_path + "hand_classes.pkl")

    def get_class_df_for_image_ids(self, _image_ids):
        _df = self.load_to_df()
        _df = _df[_df.index.isin(_image_ids)]
        return _df

    # Instead of compressed representation of left, dorsal, male, with_accessories: We here return the uncompressed
    # representation with left, right, dorsal, palmer, male, female, with_accessories, without_accessories
    def get_class_df_for_image_ids_extrapolated(self, _image_ids):
        _df = self.load_to_df_extrapolated()
        _df = _df[_df.index.isin(_image_ids)]
        return _df

    # Prepare the 8-column df from 4-column df
    def load_to_df_extrapolated(self):
        _df = self.load_to_df()
        ex = {
            0: [0, 1],
            1: [1, 0]
        }
        data = []
        for index, row in _df.iterrows():
            data.append(list(itertools.chain(*[ex[row[i]] for i in row])))

        ex_df = pd.DataFrame(data, index=_df.index)
        return ex_df

    def get_dorsal_image_ids(self):
        df = self.load_to_df()
        return df[df[1] == 1].index

    def get_palmer_image_ids(self):
        df = self.load_to_df()
        return df[df[1] == 0].index


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_2/config.json"
    config_provider = ConfigProvider(config_path)
    info_path = config_provider.get_info_path()
    storage_path = config_provider.get_storage_path()
    class_store = ClassStore(storage_path, info_path)

    df = class_store.load_to_df()
    print df.head()

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

    info = "male"

    image_factory = ImageFactory(config_provider.get_image_base_path())
    image_ids = [image_factory.get_image_id_from_path(ip) for ip in image_factory.get_image_path_list()]
    print image_ids

    df = class_store.get_class_df_for_image_ids(image_ids)

    print df
