import numpy as np

from phase_3.compare.similarity_function import EuclideanSimilarityFunction
from phase_3.config.config_provider import ConfigProvider
from phase_3.plot.image_plot import ImagePlot
from phase_3.store.feature_store import FeatureStorage, ModelType


class LSHIndexer:

    def __init__(self, _df, _k, _L):
        self.df = _df
        self.k = _k
        self.L = _L
        self.index_table = {}
        data = self.df.to_numpy()
        self.W = [np.random.randn(len(data[0]), self.k) for l in range(self.L)]
        self.sim_function = EuclideanSimilarityFunction(self.df)

    def get_index_df(self):
        data = self.df.to_numpy()

        reverse_index = {}
        for image_id in self.df.index:
            reverse_index[image_id] = ""

        for l in range(self.L):
            random_vectors = self.W[l]
            image_indices = np.dot(data, random_vectors) >= 0
            image_indices = ["".join(["1" if bit else "0" for bit in bin_index]) for bin_index in image_indices]

            for image_id, bin_index in zip(self.df.index, image_indices):
                reverse_index[image_id] = reverse_index[image_id] + \
                                          ("," if reverse_index[image_id] != "" else "") + \
                                          bin_index

        for key, index_value in reverse_index.items():
            if index_value in self.index_table:
                self.index_table[index_value].append(key)
            else:
                self.index_table[index_value] = [key]

        return self

    def get_t_similar(self, image_id, t):
        image_d = self.df.iloc[image_id, :].to_numpy()
        index = ""
        for l in range(self.L):
            random_vectors = self.W[l]
            image_indices = np.dot(image_d, random_vectors) >= 0
            image_indices = "".join(["1" if bit else "0" for bit in image_indices])
            index = index + ("," if index != "" else "") + image_indices

        result = []
        for key in self.index_table.keys():
            if sum([1 for x, y in zip(key.split(","), index.split(",")) if x == y]) > 0:
                result.extend(self.index_table[key])

        compared_images = self.get_most_similar(image_id, result)

        print "Compared with {0} images".format(len(compared_images))

        return compared_images[:t]

    def get_most_similar(self, image_id, _other_image_ids):
        similarities = [{"oid": oid, "s": self.sim_function.get_similarity(image_id, oid)} for oid in _other_image_ids]
        similarities.sort(key=lambda k: k['s'], reverse=True)
        return [s["oid"] for s in similarities]


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config.json"
    config_provider = ConfigProvider(config_path)

    storage_path = config_provider.get_storage_path_train()
    feature_storage = FeatureStorage(storage_path)
    df = feature_storage.load_to_df(ModelType.LBP)

    lsh = LSHIndexer(_df=df, _k=10, _L=10)
    table = lsh.get_index_df().index_table
    query_image_id = 415

    '''
    total = 0
    for i in table:
        print "{0}: {1}".format(i, len(table[i]))
        total = total + len(table[i])
    print total
    '''

    other_image_ids = lsh.get_t_similar(query_image_id, 10)
    print other_image_ids

    image_plot = ImagePlot(query_image_id, config_provider.get_image_base_path_train())
    plt = image_plot.plot_comparison(other_image_ids)
    plt.show()
