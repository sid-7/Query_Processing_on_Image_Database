from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem

from driver import *
from phase_3.plot.image_plot import ImagePlot
from phase_3.classifier.classifiers import do_classify
from phase_3.classifier.latent_classifier import LatentClassifier
from phase_3.clustering.clustering import KMeansClustering, Orientation
from phase_3.config.config_provider import ConfigProvider
from phase_3.indexing.lshindexer import LSHIndexer
from phase_3.mapping.image_factory import ImageFactory
from phase_3.pagerank.create_similarity_matrix import create_similarity
from phase_3.pagerank.ppr import PPR
from phase_3.plot.cluster_plot import ClusterPlot
from phase_3.reduction.feature_reduction import PCAReduction
from phase_3.relevance.relevance_feedback import RelevanceFeedback
from phase_3.store.class_store import ClassStore
from phase_3.store.feature_store import FeatureStorage

from warnings import filterwarnings

filterwarnings('ignore')

menu = ConsoleMenu("CSE515 Phase 3 Console")

task_config = {
    1: ConfigProvider(
        "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_1.json"),
    2: ConfigProvider(
        "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_2.json"),
    3: ConfigProvider(
        "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_3.json"),
    4: ConfigProvider(
        "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_4.json"),
    5: ConfigProvider(
        "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_5.json"),
    6: ConfigProvider(
        "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/phase_3/config_task_6.json"),
}

task_model = {
    1: ModelType.HOG,
    2: ModelType.HOG,
    3: ModelType.LBP,
    4: ModelType.COLOR_MOMENTS,
    5: ModelType.LBP,
    6: ModelType.LBP
}


def do_extract():
    task_id = int(raw_input("Select from [1, 2, 3, 4, 5, 6]: "))
    feature_storage = FeatureStorage(task_config[task_id].get_storage_path_train())
    image_factory = ImageFactory(task_config[task_id].get_image_base_path_train())
    do_create(feature_storage, task_model[task_id], image_factory, train=True)

    feature_storage = FeatureStorage(task_config[task_id].get_storage_path_test())
    image_factory = ImageFactory(task_config[task_id].get_image_base_path_test())
    do_create(feature_storage, task_model[task_id], image_factory, train=False)


def do_task_1():
    test_image_factory = ImageFactory(task_config[1].get_image_base_path_test())

    k = int(raw_input("Enter k: "))
    latent_classifier = LatentClassifier(_storage_path_train=task_config[1].get_storage_path_train(),
                                         _storage_path_test=task_config[1].get_storage_path_test(),
                                         _info_path=task_config[1].get_info_path(),
                                         _model_type=ModelType.HOG)

    image_ids = [test_image_factory.get_image_id_from_path(ip) for ip in test_image_factory.get_image_path_list()]
    _dorsal_image_ids = latent_classifier.class_storage.get_dorsal_image_ids()

    correct = 0
    output = {}
    for image_id in tqdm(image_ids):
        actual = "dorsal" if image_id in _dorsal_image_ids else "palmer"
        pred = latent_classifier.classify_image(_k=k, _image_id=int(image_id))
        output[image_id] = pred
        if actual == pred:
            correct = correct + 1

    print "Accuracy: {0}".format(float(correct) / len(image_ids))
    import pandas
    o_df = pandas.DataFrame(data=output.values(), columns=["orientation"], index=output.keys())
    print(o_df.to_string())


def do_task_2():
    np.seterr(divide='ignore', invalid='ignore')

    c = int(raw_input("Enter c: "))

    cluster_plot = ClusterPlot(task_config[2].get_image_base_path_train())

    kmeans_clustering = KMeansClustering(_storage_path_train=task_config[2].get_storage_path_train(),
                                         _storage_path_test=task_config[2].get_storage_path_test(),
                                         _info_path=task_config[2].get_info_path(),
                                         _image_base_path_train=task_config[2].get_image_base_path_train(),
                                         _model_type=ModelType.HOG)

    cm_dorsal = kmeans_clustering.get_cluster_memberships(c=c, orientation=Orientation.DORSAL)
    cc_dorsal = kmeans_clustering.cluster_centers
    cluster_plot.plot_memberships(cm_dorsal)

    raw_input("Continue? (y/n)")

    cm_palmer = kmeans_clustering.get_cluster_memberships(c=c, orientation=Orientation.PALMER)
    cc_palmer = kmeans_clustering.cluster_centers
    cluster_plot.plot_memberships(cm_palmer)

    image_factory_task_2 = ImageFactory(task_config[2].get_image_base_path_test())
    test_image_ids = \
        [image_factory_task_2.get_image_id_from_path(ip) for ip in image_factory_task_2.get_image_path_list()]
    _dorsal_image_ids = kmeans_clustering.class_storage.get_dorsal_image_ids()

    Y_pred = kmeans_clustering.get_predict_class(cc_dorsal, cc_palmer)
    correct = 0
    output = {}
    for i, image_id in enumerate(test_image_ids):
        actual = "dorsal" if image_id in _dorsal_image_ids else "palmer"
        pred = Y_pred[i]
        output[image_id] = pred
        if actual == pred:
            correct = correct + 1

    print "Accuracy: {0}".format(float(correct) / len(test_image_ids))
    import pandas
    o_df = pandas.DataFrame(data=output.values(), columns=["orientation"], index=output.keys())
    print(o_df.to_string())


def do_task_3():
    k = int(raw_input("Enter k: "))
    K = int(raw_input("Enter K: "))
    seeds = [int(seed) for seed in raw_input("Enter seeds: ").split(",")]
    feature_storage = FeatureStorage(task_config[3].get_storage_path_train())
    df = feature_storage.load_to_df(ModelType.LBP)
    create_similarity(df)
    ppr = PPR()
    similar = ppr.compute(seeds, K, k)
    other_image_ids = [s["other_image_id"] for s in similar]
    image_plot = ImagePlot(other_image_ids[0], task_config[3].get_image_base_path_train())
    plt = image_plot.plot_comparison(other_image_ids[0:])
    plt.show()
    import pandas
    o_df = pandas.DataFrame(data=similar, columns=["other_image_id", "score"])
    print(o_df.to_string())


def do_task_4():
    classifier = raw_input("Enter from [svm, dt, ppr]: ")
    feature_storage_train = FeatureStorage(task_config[4].get_storage_path_train())
    feature_storage_test = FeatureStorage(task_config[4].get_storage_path_test())
    class_storage = ClassStore(task_config[4].get_storage_path_train(), task_config[4].get_info_path())
    model_type = ModelType.COLOR_MOMENTS

    X_train = feature_storage_train.load_to_df(model_type)
    y_train = class_storage.get_class_df_for_image_ids(X_train.index)[1]

    X_test = feature_storage_test.load_to_df(model_type)
    y_test = class_storage.get_class_df_for_image_ids(X_test.index)[1]

    if classifier == "svm":
        k = 100
        reduction = PCAReduction(task_config[4].get_storage_path_train(), model_type, task_config[4].get_info_path())
        reduction.df = X_train
        X_train = reduction.get_u(k)

        reduction.df = X_test
        X_test = reduction.get_u(k)

    accuracy = do_classify(classifier, X_train, X_test, y_train, y_test)
    print "Accuracy: {0}".format(accuracy)


def do_task_5():
    L = int(raw_input("Enter L: "))
    k = int(raw_input("Enter k: "))

    feature_storage = FeatureStorage(task_config[5].get_storage_path_train())
    df = feature_storage.load_to_df(ModelType.LBP)
    lsh = LSHIndexer(_df=df, _k=k, _L=L)
    table = lsh.get_index_df().index_table
    print table

    t = int(raw_input("Enter t: "))
    image_id = int(raw_input("Enter image ID: "))

    other_image_ids = lsh.get_t_similar(image_id, t)
    print other_image_ids

    image_plot = ImagePlot(image_id, task_config[5].get_image_base_path_train())
    plt = image_plot.plot_comparison(other_image_ids)
    plt.show()


def do_task_6():
    model = raw_input("Enter from [svm, dt, ppr, prob]: ")
    L = int(raw_input("Enter L: "))
    k = int(raw_input("Enter k: "))
    t = int(raw_input("Enter t: "))
    image_id = int(raw_input("Enter image ID: "))

    rel_feedback = RelevanceFeedback(task_config[6], task_model[6])
    rel_feedback.driver(k, L, t, image_id, model)


extract_item = FunctionItem("Extract Features", do_extract)
task_1_item = FunctionItem("Task 1", do_task_1)
task_2_item = FunctionItem("Task 2", do_task_2)
task_3_item = FunctionItem("Task 3", do_task_3)
task_4_item = FunctionItem("Task 4", do_task_4)
task_5_item = FunctionItem("Task 5", do_task_5)
task_6_item = FunctionItem("Task 6", do_task_6)

menu.append_item(task_1_item)
menu.append_item(task_2_item)
menu.append_item(task_3_item)
menu.append_item(task_4_item)
menu.append_item(task_5_item)
menu.append_item(task_6_item)
menu.append_item(extract_item)

menu.show()
