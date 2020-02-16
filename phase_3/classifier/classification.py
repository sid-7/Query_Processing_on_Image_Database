import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from phase_3.config.config_provider import ConfigProvider
from phase_3.reduction.feature_reduction import SVDReduction
from phase_3.store.feature_store import ModelType


# Classifier base class
class Classifier:

    def __init__(self, reduction, model_type, class_names, k):
        config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_2/config.json"
        config_provider = ConfigProvider(config_path)
        self.X = pd.DataFrame()
        self.Y = []
        for class_name in class_names:
            reducer = reduction(storage_path=config_provider.get_storage_path(),
                                info_path=config_provider.get_info_path(),
                                model_type=model_type,
                                info=class_name)

            reduced_features, Y = reducer.get_x_y(k=k)
            self.X = self.X.append(reduced_features)
            self.Y.extend(Y)

    def get_training_data(self):
        X = self.X
        Y = self.Y
        return X, Y

    # Override differently for every classifier
    def get_model(self, X, Y):
        return

    # Get target class predicts the class for image_id
    def get_target_class(self, image_id):
        x = np.asarray(self.X.loc[self.X.index == image_id])
        x_i = list(self.X.index).index(image_id)
        y = self.Y[x_i]
        X = self.X.loc[self.X.index != image_id]
        Y = [y for i, y in enumerate(self.Y) if i != x_i]
        model = self.get_model(X, Y)
        y_pred = model.predict(x)
        return y_pred

    # This method will convert the target class into a readable name
    @staticmethod
    def get_target_class_name(target_class, input_class_type):
        return {
            "male": {0: "female", 1: "male"},
            "female": {0: "female", 1: "male"},
            "with-accessories": {0: "without-accessories", 1: "with-accessories"},
            "without-accessories": {0: "without-accessories", 1: "with-accessories"},
            "left-hand": {0: "right-hand", 1: "left-hand"},
            "right-hand": {0: "right-hand", 1: "left-hand"},
            "dorsal": {0: "palmer", 1: "dorsal"},
            "palmer": {0: "palmer", 1: "dorsal"}
        }[input_class_type][target_class]


class SVMClassifier(Classifier):

    def get_model(self, X, Y):
        clf = svm.SVC(gamma='scale')
        clf.fit(X, Y)
        return clf


class KNNClassifier(Classifier):

    def get_model(self, X, Y):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(X, Y)
        return neigh


class LRClassifier(Classifier):

    def get_model(self, X, Y):
        clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
        clf.fit(X, Y)
        return clf


if __name__ == "__main__":
    classifier = LRClassifier(reduction=SVDReduction,
                              model_type=ModelType.COLOR_MOMENTS,
                              class_names=["male", "female"],
                              k=10)

    classifier.get_target_class(12)
