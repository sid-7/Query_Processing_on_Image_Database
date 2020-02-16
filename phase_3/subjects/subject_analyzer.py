import pandas as pd

from phase_2.compare.similarity_function import EuclideanSimilarityFunction
from phase_2.config.config_provider import ConfigProvider
from phase_2.mapping.image_factory import ImageFactory
from phase_2.mapping.vector_mapping import *
from phase_2.reduction.feature_reduction import NMFReduction
from phase_2.store.feature_store import ModelType

lbp_vector_mapping = LbpVectorMapping()
cm_vector_mapping = ColorMomentsVectorMapping()
hog_vector_mapping = HogVectorMapping()
sift_vector_mapping = SiftVectorMapping()


class SubjectAnalyzer:

    def __init__(self, _info_path, _image_base_path, _model_type, _test_subjects=None):
        self.test_subjects = _test_subjects
        self.model_type = _model_type
        self.image_factory = ImageFactory(_image_base_path)
        # Get image_names of all the images in image_base_path
        self.image_names = ["Hand_" + str(self.image_factory.get_image_id_from_path(image_path)).zfill(7) + ".jpg" for
                            image_path in self.image_factory.get_image_path_list()]
        self.df = pd.read_csv(_info_path, sep=",")

        # Get subjects in these images
        self.subjects = self.df[self.df["imageName"].isin(self.image_names)]["id"].unique()

    def get_feature_df(self):
        final_df = pd.DataFrame(columns=["dr_a", "dr_wa", "dl_a", "dl_wa", "pr_a", "pr_wa", "pl_a", "pl_wa"])
        # For each subject get the image_ids
        for subject_id in self.test_subjects if self.test_subjects is not None else self.subjects:
            s_df = self.df[self.df["id"] == subject_id]

            # Find average feature vector for each of the 8 orientations
            dr_a = s_df[(s_df["aspectOfHand"] == "dorsal right") & (s_df["accessories"] == 1)]["imageName"].unique()
            mean_fv_dr_a = self.get_average_feature_model(dr_a)

            dr_wa = s_df[(s_df["aspectOfHand"] == "dorsal right") & (s_df["accessories"] == 0)]["imageName"].unique()
            mean_fv_dr_wa = self.get_average_feature_model(dr_wa)

            dl_a = s_df[(s_df["aspectOfHand"] == "dorsal left") & (s_df["accessories"] == 1)]["imageName"].unique()
            mean_fv_dl_a = self.get_average_feature_model(dl_a)

            dl_wa = s_df[(s_df["aspectOfHand"] == "dorsal left") & (s_df["accessories"] == 0)]["imageName"].unique()
            mean_fv_dl_wa = self.get_average_feature_model(dl_wa)

            pr_a = s_df[(s_df["aspectOfHand"] == "palmar right") & (s_df["accessories"] == 1)]["imageName"].unique()
            mean_fv_pr_a = self.get_average_feature_model(pr_a)

            pr_wa = s_df[(s_df["aspectOfHand"] == "palmar right") & (s_df["accessories"] == 0)]["imageName"].unique()
            mean_fv_pr_wa = self.get_average_feature_model(pr_wa)

            pl_a = s_df[(s_df["aspectOfHand"] == "palmar left") & (s_df["accessories"] == 1)]["imageName"].unique()
            mean_fv_pl_a = self.get_average_feature_model(pl_a)

            pl_wa = s_df[(s_df["aspectOfHand"] == "palmar left") & (s_df["accessories"] == 0)]["imageName"].unique()
            mean_fv_pl_wa = self.get_average_feature_model(pl_wa)

            # Final subject_id will have 8 feature vectors (in single row for a subject)
            final_df.loc[subject_id] = [mean_fv_dr_a] + [mean_fv_dr_wa] + [mean_fv_dl_a] + [mean_fv_dl_wa] + \
                                       [mean_fv_pr_a] + [mean_fv_pr_wa] + [mean_fv_pl_a] + [mean_fv_pl_wa]
        return final_df

    # Averages out the feature vectors of multiple images for same orientation
    def get_average_feature_model(self, _image_names):
        result = []
        for image_id in [self.image_factory.get_image_id_from_path(image_name) for image_name in _image_names]:
            image_path = self.image_factory.get_image_path_from_id(image_id)
            if image_path in self.image_factory.get_image_path_list():
                if self.model_type == ModelType.COLOR_MOMENTS:
                    cm_from_image = cm_vector_mapping.get_color_moments_from_image(image_path)
                    result.append(cm_from_image)
                elif self.model_type == ModelType.LBP:
                    lbp_from_image = lbp_vector_mapping.get_lbp_from_image(image_path)
                    result.append(lbp_from_image)
                elif self.model_type == ModelType.HOG:
                    hog_from_image = hog_vector_mapping.get_hog_from_image(image_path)
                    result.append(hog_from_image)
                elif self.model_type == ModelType.SIFT:
                    sift_from_image = sift_vector_mapping.get_sift_from_image(image_path)
                    result.append(sift_from_image)

        if len(result) > 0:
            result = np.mean(np.array(result), axis=0)

        return result

    # Get subject-subject similarity matrix. Match the orientation feature vectors (if both the subjects have the data)
    # and average out the distances
    def get_sss_matrix(self):
        feature_df = self.get_feature_df()
        similarity_function = EuclideanSimilarityFunction(None)
        sss_matrix = pd.DataFrame(index=self.subjects, columns=self.subjects)
        for subject_id in self.subjects:
            for other_subject_id in [subject for subject in self.subjects]:
                distances = []
                for orientation in ["dr_a", "dr_wa", "dl_a", "dl_wa", "pr_a", "pr_wa", "pl_a", "pl_wa"]:
                    x = np.array(feature_df.loc[subject_id, orientation])
                    y = np.array(feature_df.loc[other_subject_id, orientation])
                    if (x.shape != (0,)) and (y.shape != (0,)):
                        distances.append(similarity_function.l2_dist(x, y))
                average_distance = sum(distances) / len(distances) if len(distances) > 0 else 9999999.99
                # Similarity = 1 / 1 + distance
                similarity = 1 / (1 + average_distance)
                sss_matrix.loc[subject_id][other_subject_id] = similarity
        return sss_matrix

    # Get all similar subjects sorting the sss_matrix row in ascending order and slicing [1:4] (first one is self)
    def get_similar_subjects(self, subject_id):
        sss_matrix = self.get_sss_matrix()
        distances = [{"other_subject_id": column, "distance": sss_matrix.loc[subject_id, column]} for column in list(sss_matrix.columns)]
        distances.sort(key=lambda n: n["distance"], reverse=True)
        return distances[1:4]


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_2/config.json"
    config_provider = ConfigProvider(config_path)
    image_base_path = config_provider.get_image_base_path()
    storage_path = config_provider.get_storage_path()
    output_path = config_provider.get_output_path()
    info_path = config_provider.get_info_path()
    subject_analyzer = SubjectAnalyzer(_info_path=info_path,
                                       _image_base_path=image_base_path,
                                       _model_type=ModelType.COLOR_MOMENTS)

    # print subject_analyzer.get_similar_subjects(subject_id=0)

    reduction = NMFReduction(storage_path=config_provider.get_storage_path(),
                             info_path=config_provider.get_info_path(),
                             model_type=ModelType.COLOR_MOMENTS)

    reduction.df = subject_analyzer.get_sss_matrix()

    data_semantics = reduction.get_data_latent_semantics(k=10)
    for l in reduction.get_latent_data_term_weight_pairs(data_semantics):
        print l
