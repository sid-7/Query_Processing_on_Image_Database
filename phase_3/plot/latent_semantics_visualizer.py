import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from phase_2.config.config_provider import ConfigProvider
from phase_2.reduction.feature_reduction import SVDReduction
from phase_2.store.feature_store import ModelType

import os
import tqdm


class DataSemanticsPlot:

    def __init__(self, image_base_path):
        self.image_base_path = image_base_path

    # image_id_scores is a list of tuples (image_id, score) (sorted desc)
    def plot_comparison(self, image_id_scores):
        n = len(image_id_scores)

        fig = plt.figure(figsize=(n * 1.1, 1))

        for i in range(0, n):
            image_file_name = self.image_base_path + "Hand_" + str(image_id_scores[i][0]).zfill(7) + ".jpg"
            img = mpimg.imread(image_file_name)
            ax = fig.add_subplot(1, n + 1, i + 1)
            score = str(image_id_scores[i][1])[0:8]
            ax.title.set_text(score)
            plt.imshow(img)

        return plt

    # Plot for each term_weight_pairs set (for all the semantics)
    def plot_semantics(self, _output_path, _term_weight_pairs):
        query_id = str(datetime.datetime.now().time())

        for i, twp in tqdm.tqdm(enumerate(_term_weight_pairs)):
            _plt = self.plot_comparison(twp)
            self.save_plot(_plt, i, _output_path, query_id)

    # Saving the plot to a file with timestamp as query ID
    @staticmethod
    def save_plot(_plt, _i, _output_path, _query_id):
        output_directory = _output_path + "/query_" + _query_id
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_image_path = output_directory + "/output_" + str(_i) + ".png"
        _plt.savefig(output_image_path)


class FeatureSemanticsPlot:

    def __init__(self, image_base_path):
        self.image_base_path = image_base_path

    # image_id_scores is a list of tuples (image_id, score) (sorted desc)
    def plot_comparison(self, image_id_scores):
        n = len(image_id_scores)

        fig = plt.figure(figsize=(n * 1.1, 1))

        for i in range(0, n):
            image_file_name = self.image_base_path + "Hand_" + str(image_id_scores[i]["image_id"]).zfill(7) + ".jpg"
            img = mpimg.imread(image_file_name)
            ax = fig.add_subplot(1, n + 1, i + 1)
            score = ('%.4f' % image_id_scores[i]["dot"])
            ax.title.set_text(score)
            plt.imshow(img)

        return plt

    # Plot just the feature_weight_pairs for each semantic
    def plot_semantics(self, _output_path, _term_weight_pairs):
        query_id = str(datetime.datetime.now().time())
        _plt = self.plot_comparison(_term_weight_pairs)
        self.save_plot(_plt, _output_path, query_id)

    @staticmethod
    def save_plot(_plt, _output_path, _query_id):
        output_directory = _output_path + "/query_" + _query_id
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        output_image_path = output_directory + "/output_feature_semantics.png"
        _plt.savefig(output_image_path)


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_2/config.json"
    config_provider = ConfigProvider(config_path)

    output_path = config_provider.get_output_path()

    reduction = SVDReduction(storage_path=config_provider.get_storage_path(),
                             info_path=config_provider.get_info_path(),
                             model_type=ModelType.COLOR_MOMENTS)

    data_latent_semantics = reduction.get_data_latent_semantics(k=10)

    term_weight_pairs = reduction.get_latent_data_term_weight_pairs(data_latent_semantics)

    # image_plot = DataSemanticsPlot(config_provider.get_image_base_path())

    # image_plot.plot_semantics(output_path, term_weight_pairs)

    image_plot = FeatureSemanticsPlot(config_provider.get_image_base_path())

    image_plot.plot_semantics(output_path, reduction.get_feature_latent_semantics(k=10))
