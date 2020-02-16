import json

import pandas as pd
from enum import Enum


class ModelType(Enum):
    LBP = "lbp"
    COLOR_MOMENTS = "cm"
    HOG = "hog"
    SIFT = "sift"


class FeatureStorage:

    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.switcher_json = {
            ModelType.LBP: "lbp.json",
            ModelType.COLOR_MOMENTS: "color_moments.json",
            ModelType.HOG: "hog.json",
            ModelType.SIFT: "sift.json",
        }

        self.switcher_pkl = {
            ModelType.LBP: "lbp.pkl",
            ModelType.COLOR_MOMENTS: "color_moments.pkl",
            ModelType.HOG: "hog.pkl",
            ModelType.SIFT: "sift.pkl",
        }

        self.switcher_distances = {
            ModelType.LBP: "lbp.distances.json",
            ModelType.COLOR_MOMENTS: "color_moments.distances.json",
            ModelType.HOG: "hog.distances.json",
            ModelType.SIFT: "sift.distances.json",
        }

    def clear_model_storage(self, model_type):
        open(self.storage_path + self.switcher_json.get(model_type), "w").close()

    def store_feature_set(self, model_type, feature_vectors):
        db_file = self.storage_path + self.switcher_json.get(model_type)
        with open(db_file, 'w') as f:
            json.dump(feature_vectors, f)
        self.store_to_pkl(model_type)

    def load_to_df(self, model_type):
        db_file = self.storage_path + self.switcher_pkl.get(model_type)
        df = pd.read_pickle(db_file)
        return df

    def store_to_pkl(self, model_type):
        db_file = self.storage_path + self.switcher_json.get(model_type)
        df = pd.read_json(db_file)
        df = df.T
        df.to_pickle(self.storage_path + self.switcher_pkl.get(model_type))

    def store_distances(self, model_type, distances_json):
        db_file = self.storage_path + self.switcher_json.get(model_type)
        with open(db_file, 'w') as f:
            json.dump(distances_json, f)
