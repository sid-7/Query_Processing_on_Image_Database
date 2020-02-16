import pandas as pd

from phase_3.compare.similarity_function import EuclideanSimilarityFunction
from phase_3.config.config_provider import ConfigProvider
from phase_3.indexing.lshindexer import LSHIndexer
from phase_3.pagerank.create_similarity_matrix import create_similarity
from phase_3.pagerank.ppr import PPR
from phase_3.plot.image_plot import ImagePlot
from phase_3.store.class_store import ClassStore
from phase_3.store.feature_store import FeatureStorage, ModelType

from warnings import filterwarnings

filterwarnings('ignore')


class RelevanceFeedback:

    def __init__(self, _config_provider, _model_type):
        self.config_provider = _config_provider
        self.feature_storage = FeatureStorage(_config_provider.get_storage_path_train())
        self.df = self.feature_storage.load_to_df(_model_type)
        self.model_switcher = {
            "svm": self.do_svm,
            "dt": self.do_dt,
            "ppr": self.do_ppr,
            "prob": self.do_prob
        }

    def get_initial_results(self, _k, _L, _t, _image_id):
        _lsh = LSHIndexer(_df=self.df, _k=_k, _L=_L)
        _lsh.get_index_df()
        _other_image_ids = _lsh.get_t_similar(_image_id, _t)
        return _other_image_ids

    def get_training_data(self, _relevant, _irrelevant):
        sim_function = EuclideanSimilarityFunction(self.df)
        _train_relevant = []
        for _relevant_image in _relevant:
            _train_relevant.extend(
                [d["other_image_id"] for d in sim_function.get_image_distances(_relevant_image)[:100]])
            _train_relevant.extend(_relevant)

        _train_irrelevant = []
        for _irrelevant_image in _irrelevant:
            _train_irrelevant.extend(
                [d["other_image_id"] for d in sim_function.get_image_distances(_irrelevant_image)[:100]])
            _train_irrelevant.extend(_irrelevant)

        _train_irrelevant = [i for i in _train_irrelevant if i not in _train_relevant]

        _X_train = self.df.ix[set(_train_relevant + _train_irrelevant)]
        _rs = len([i for i in _X_train.index if i in set(_train_relevant)])
        _irs = len([i for i in _X_train.index if i in set(_train_irrelevant)])
        _y_train = pd.Series([1 for i in range(_rs)] + [0 for i in range(_irs)], index=_X_train.index)

        return _X_train, _y_train

    def do_svm(self, _X_train, _y_train):
        from sklearn.svm import LinearSVC
        clf = LinearSVC(random_state=0, tol=1e-5)
        clf.fit(_X_train.to_numpy(), _y_train)
        y_pred = clf.predict(self.df.to_numpy())
        adf = self.df.ix[[image for i, image in enumerate(self.df.index) if y_pred[i] == 1]]
        return adf

    def do_dt(self, _X_train, _y_train):
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(_X_train.to_numpy(), _y_train)
        y_pred = clf.predict(self.df.to_numpy())
        adf = self.df.ix[[image for i, image in enumerate(self.df.index) if y_pred[i] == 1]]
        return adf

    def do_ppr(self, _X_train, _y_train):
        create_similarity(_X_train)
        seeds = list(_X_train.index[:list(_y_train).count(1)])
        ppr = PPR()
        sim = ppr.compute(list(seeds), _K=len(_X_train), _k=5)
        adf = self.df.ix[[i for i in self.df.index if i in [s["other_image_id"] for s in sim]]]
        return adf

    def do_prob(self, _X_train, _y_train):
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(_X_train.to_numpy(), _y_train)
        y_pred = clf.predict(self.df.to_numpy())
        adf = self.df.ix[[image for i, image in enumerate(self.df.index) if y_pred[i] == 1]]
        return adf

    @staticmethod
    def get_new_results(adf, _k, _L, _t, _image_id):
        lsh = LSHIndexer(_df=adf, _k=_k, _L=_L)
        lsh.get_index_df()
        other_image_ids = lsh.get_t_similar(_image_id, _t)
        return other_image_ids

    def driver(self, _k, _L, _t, _image_id, _model):
        itr = 0
        while True:
            try:
                result = self.get_initial_results(_k, _L, _t, _image_id)

                print result

                image_plot = ImagePlot(_image_id, self.config_provider.get_image_base_path_train())
                plt = image_plot.plot_comparison(result)
                plt.show()

                if itr > 0:
                    a = "stop" if raw_input("Stop? (y/n)") == "y" else "go"
                    if a == "stop":
                        break

                _relevant = raw_input("Enter relevant: ")
                _relevant = [int(i) for i in _relevant.split(",")]

                _irrelevant = raw_input("Enter irrelevant: ")
                _irrelevant = [int(i) for i in _irrelevant.split(",")]

                X_train, y_train = self.get_training_data(_relevant, _irrelevant)

                adf = self.model_switcher[_model](X_train, y_train)

                _image_id = adf.index[0] if _model == "ppr" else _image_id

                itr = itr + 1

                self.df = adf

            except IndexError or KeyError:

                print "No more iterations possible"
                break


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_6.json"
    config_provider = ConfigProvider(config_path)

    class_storage = ClassStore(config_provider.get_storage_path_train(), config_provider.get_info_path())

    L = 10
    k = 10
    t = 20
    image_id = 674

    rel_feedback = RelevanceFeedback(config_provider, ModelType.LBP)
    rel_feedback.driver(k, L, t, image_id, "svm")
