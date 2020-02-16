from phase_1.sandeepkunichi.store.feature_store import FeatureStorage, ModelType


# Pickle the HOG
storage_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/store/"
feature_storage = FeatureStorage(storage_path)
feature_storage.store_to_pkl(ModelType.HOG)


