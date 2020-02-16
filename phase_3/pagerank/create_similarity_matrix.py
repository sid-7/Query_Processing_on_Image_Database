import multiprocessing
from contextlib import contextmanager

import numpy
import pandas
from tqdm import tqdm

from phase_3.config.config_provider import ConfigProvider
from phase_3.store.feature_store import FeatureStorage, ModelType


def sim(_input):
    return _input["oid"], 1 / (1 + numpy.sqrt(numpy.sum((_input["x"] - _input["y"]) ** 2)))


@contextmanager
def pool_context(*args, **kwargs):
    _pool = multiprocessing.Pool(*args, **kwargs)
    yield _pool
    _pool.terminate()


def create_similarity(_df):
    s_matrix = pandas.DataFrame(index=_df.index, columns=_df.index)
    for image_id in tqdm(_df.index):
        x = _df.loc[image_id, :]
        with pool_context(processes=10) as pool:
            ip = [{"x": x, "y": _df.loc[other_image_id, :], "oid": other_image_id} for other_image_id in _df.index]
            s_matrix.loc[image_id] = [sim_score for other_image_id, sim_score in pool.imap(sim, ip)]

    output_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/sim.pkl"
    s_matrix.to_pickle(output_path)


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_3.json"
    config_provider = ConfigProvider(config_path)

    storage_path_train = config_provider.get_storage_path_train()
    model_type = ModelType.LBP

    feature_storage_train = FeatureStorage(storage_path_train)
    df = feature_storage_train.load_to_df(model_type)

    create_similarity(df)
