import multiprocessing
from contextlib import contextmanager

from compare.similarity_function import *
from mapping.vector_mapping import *
from store.feature_store import ModelType

lbp_vector_mapping = LbpVectorMapping()
cm_vector_mapping = ColorMomentsVectorMapping()
hog_vector_mapping = HogVectorMapping()
sift_vector_mapping = SiftVectorMapping()


@contextmanager
def pool_context(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()


def do_cm_task(_image_path):
    _image_id = _image_path[1].get_image_id_from_path(_image_path)
    cm_from_image = cm_vector_mapping.get_color_moments_from_image(_image_path[0])
    return _image_id, cm_from_image.tolist()


def do_lbp_task(_image_path):
    _image_id = _image_path[1].get_image_id_from_path(_image_path)
    lbp_from_image = lbp_vector_mapping.get_lbp_from_image(_image_path[0])
    return _image_id, lbp_from_image.tolist()


def do_hog_task(_image_path):
    _image_id = _image_path[1].get_image_id_from_path(_image_path)
    hog_from_image = hog_vector_mapping.get_hog_from_image(_image_path[0])
    return _image_id, hog_from_image.tolist()


def do_sift_task(_image_path):
    _image_id = _image_path[1].get_image_id_from_path(_image_path)
    sift_from_image = sift_vector_mapping.get_sift_from_image(_image_path[0])
    return _image_id, sift_from_image.tolist()


def do_parallel_train(task, image_factory):
    feature_vectors = {}
    with pool_context(processes=10) as pool:
        _image_paths = [(ip, image_factory) for ip in image_factory.get_image_path_list()]
        for _image_id, _image_feature_vector in list(tqdm(pool.imap(task, _image_paths), total=len(_image_paths))):
            feature_vectors[_image_id] = _image_feature_vector
    return feature_vectors


def do_parallel_test(task, image_factory):
    feature_vectors = {}
    with pool_context(processes=10) as pool:
        _image_paths = [(ip, image_factory) for ip in image_factory.get_image_path_list()]
        for _image_id, _image_feature_vector in list(tqdm(pool.imap(task, _image_paths), total=len(_image_paths))):
            feature_vectors[_image_id] = _image_feature_vector
    return feature_vectors


def do_create(_feature_storage, _model_type, image_factory, train):
    print "Creating " + _model_type + " features ({0}):".format("Train" if train else "Test")
    do_parallel = do_parallel_train if train else do_parallel_test

    if _model_type == ModelType.COLOR_MOMENTS:
        _feature_storage.clear_model_storage(ModelType.COLOR_MOMENTS)
        _feature_storage.store_feature_set(ModelType.COLOR_MOMENTS, do_parallel(do_cm_task, image_factory))
    elif _model_type == ModelType.LBP:
        _feature_storage.clear_model_storage(ModelType.LBP)
        _feature_storage.store_feature_set(ModelType.LBP, do_parallel(do_lbp_task, image_factory))
    elif _model_type == ModelType.HOG:
        _feature_storage.clear_model_storage(ModelType.HOG)
        _feature_storage.store_feature_set(ModelType.HOG, do_parallel(do_hog_task, image_factory))
    elif _model_type == ModelType.SIFT:
        _feature_storage.clear_model_storage(ModelType.SIFT)
        _feature_storage.store_feature_set(ModelType.SIFT, do_parallel(do_sift_task, image_factory))
    print "Features stored in " + _feature_storage.storage_path
