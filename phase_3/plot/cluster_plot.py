import json

from phase_3.clustering.clustering import Orientation, KMeansClustering
from phase_3.config.config_provider import ConfigProvider
from phase_3.mapping.image_factory import ImageFactory
from phase_3.store.feature_store import ModelType


class ClusterPlot:

    def __init__(self, image_base_path):
        self.image_base_path = image_base_path
        self.json_path = '/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/viz/clusters.json'

    def plot_memberships(self, cluster_memberships):
        c = len(cluster_memberships.keys())
        image_factory = ImageFactory(self.image_base_path)
        viz_json = {
            "name": "clusters",
            "children": [
                {
                    "name": "Cluster {0}".format(i + 1),
                    "children": [{
                        "name": str(image_id),
                        "img": "http://localhost:8080/Hands/{0}".format(image_factory.get_image_name_from_id(image_id)),
                        "size": 40000
                    } for image_id in cluster_memberships[i]]
                } for i in range(c)
            ]
        }

        with open(self.json_path, "w") as outfile:
            json.dump(viz_json, outfile)

        print "Go to: http://localhost:8080"


if __name__ == "__main__":
    config_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_2.json"
    config_provider = ConfigProvider(config_path)

    storage_path_train = config_provider.get_storage_path_train()
    storage_path_test = config_provider.get_storage_path_test()
    image_base_path_train = config_provider.get_image_base_path_train()
    info_path = config_provider.get_info_path()

    kmeans_clustering = KMeansClustering(_storage_path_train=storage_path_train,
                                         _storage_path_test=storage_path_test,
                                         _info_path=info_path,
                                         _image_base_path_train=image_base_path_train,
                                         _model_type=ModelType.LBP)

    c = 10
    # cm_dorsal = kmeans_clustering.get_cluster_memberships(c, Orientation.DORSAL)
    cm_palmer = kmeans_clustering.get_cluster_memberships(c, Orientation.PALMER)

    cluster_plot = ClusterPlot(config_provider.get_image_base_path_train())
    # cluster_plot.plot_memberships(cm_dorsal)
    cluster_plot.plot_memberships(cm_palmer)
