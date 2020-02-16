import numpy
from numpy.linalg import norm
from skimage.measure import compare_ssim
from tqdm import tqdm


class SimilarityFunction:

    def __init__(self, df):
        self.df = df

    @staticmethod
    def extract_distance(json):
        try:
            return float(json["s"])
        except KeyError:
            return 0

    def get_similarity(self, image_id_1, image_id_2):
        pass

    def get_image_distances(self, image_id):
        distances = []
        print ("Calculating distances of images from image_id = " + str(image_id))

        for other_image_id in tqdm([x for x in self.df.index.values if x != image_id]):
            distances.append({"s": self.get_similarity(image_id, other_image_id),
                              "other_image_id": other_image_id})

        distances.sort(key=self.extract_distance, reverse=True)
        return distances


class SsimSimilarityFunction(SimilarityFunction):

    def __init__(self, df):
        SimilarityFunction.__init__(self, df)

    def get_similarity(self, image_id_1, image_id_2):
        return compare_ssim(self.df.loc[image_id_1, :], self.df.loc[image_id_2, :])


class CosineSimilarityFunction(SimilarityFunction):

    def __init__(self, df):
        SimilarityFunction.__init__(self, df)

    def get_similarity(self, image_id_1, image_id_2):
        dot_prod = numpy.dot(self.df.loc[image_id_1, :], self.df.loc[image_id_2, :])
        norms = (norm(self.df.loc[image_id_1, :]) * norm(self.df.loc[image_id_2, :]))
        return dot_prod / norms


class EuclideanSimilarityFunction(SimilarityFunction):

    def __init__(self, df):
        SimilarityFunction.__init__(self, df)

    @staticmethod
    def l2_dist(x, y):
        return numpy.sqrt(numpy.sum((x - y) ** 2))

    def get_similarity(self, image_id_1, image_id_2):
        return 0 - self.l2_dist(self.df.loc[image_id_1, :], self.df.loc[image_id_2, :])
