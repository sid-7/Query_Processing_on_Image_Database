from consolemenu import ConsoleMenu
from consolemenu.items import FunctionItem

from driver import *
from phase_2.classifier.classification import LRClassifier
from phase_2.reduction.feature_reduction import SVDReduction, PCAReduction, NMFReduction, LDAReduction
from phase_2.subjects.subject_analyzer import SubjectAnalyzer
menu = ConsoleMenu("CSE515 Phase 2 Console")

reduction_switcher = {
    "svd": SVDReduction,
    "pca": PCAReduction,
    "nmf": NMFReduction,
    "lda": LDAReduction
}

class_type_switcher = {
    "male": ["male", "female"],
    "female": ["male", "female"],
    "with-accessories": ["with-accessories", "without-accessories"],
    "without-accessories": ["with-accessories", "without-accessories"],
    "left-hand": ["left-hand", "right-hand"],
    "right-hand": ["left-hand", "right-hand"],
    "dorsal": ["dorsal", "palmer"],
    "palmer": ["dorsal", "palmer"]
}


def do_extract():
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    do_create(image_factory, cm_vector_mapping, feature_storage, model_type)


def do_task_1():
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    k = int(raw_input("Enter k: "))
    reduction = raw_input("Enter reduction technique from [svd, pca, nbp, lda]: ")
    reduction = reduction_switcher.get(reduction)(storage_path=config_provider.get_storage_path(),
                                                  info_path=config_provider.get_info_path(),
                                                  model_type=model_type)
    data_latent_semantics = reduction.get_data_latent_semantics(k=k)
    term_weight_pairs = reduction.get_latent_data_term_weight_pairs(data_latent_semantics)
    feature_latent_semantics = reduction.get_feature_latent_semantics(k=k)

    data_semantics_plot.plot_semantics(output_path, term_weight_pairs)
    feature_semantics_plot.plot_semantics(output_path, feature_latent_semantics)


def do_task_2():
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    k = int(raw_input("Enter k: "))
    reduction = raw_input("Enter reduction technique from [svd, pca, nbp, lda]: ")
    image_id = int(raw_input("Enter Image ID: "))
    m = int(raw_input("Enter m: "))
    reduction = reduction_switcher.get(reduction)(storage_path=config_provider.get_storage_path(),
                                                  info_path=config_provider.get_info_path(),
                                                  model_type=model_type)

    distances = reduction.get_k_similar_from_data_semantics(k=k, m=m, image_id=image_id)
    print distances
    plt = do_plot(image_id, distances)
    save_plot(plt, image_id, {})
    plt.show()


def do_task_3():
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    label = raw_input("Select from [left-hand, right-hand, dorsal, palmer, with-accessories, without-accessories, "
                      "male, female]: ")
    k = int(raw_input("Enter k: "))
    reduction = raw_input("Enter reduction technique from [svd, pca, nbp, lda]: ")
    reduction = reduction_switcher.get(reduction)(storage_path=config_provider.get_storage_path(),
                                                  info_path=config_provider.get_info_path(),
                                                  model_type=model_type,
                                                  info=label)
    data_latent_semantics = reduction.get_data_latent_semantics(k=k)
    term_weight_pairs = reduction.get_latent_data_term_weight_pairs(data_latent_semantics)
    feature_latent_semantics = reduction.get_feature_latent_semantics(k=k)

    data_semantics_plot.plot_semantics(output_path, term_weight_pairs)
    feature_semantics_plot.plot_semantics(output_path, feature_latent_semantics)


def do_task_4():
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    reduction = raw_input("Enter reduction technique from [svd, pca, nbp, lda]: ")
    k = int(raw_input("Enter k: "))
    label = raw_input("Select from [left-hand, right-hand, dorsal, palmer, with-accessories, without-accessories, "
                      "male, female]: ")
    image_id = int(raw_input("Enter Image ID: "))
    m = int(raw_input("Enter m: "))
    reduction = reduction_switcher.get(reduction)(storage_path=config_provider.get_storage_path(),
                                                  info_path=config_provider.get_info_path(),
                                                  model_type=model_type,
                                                  info=label)

    distances = reduction.get_k_similar_from_data_semantics(k=k, m=m, image_id=image_id)
    print distances
    plt = do_plot(image_id, distances)
    save_plot(plt, image_id, {})
    plt.show()


def do_task_5():
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    reduction = raw_input("Enter reduction technique from [svd, pca, nbp, lda]: ")
    k = int(raw_input("Enter k: "))
    label = raw_input("Select from [left-hand, right-hand, dorsal, palmer, with-accessories, without-accessories, "
                      "male, female]: ")
    image_id = int(raw_input("Enter Image ID: "))
    reduction = reduction_switcher.get(reduction)

    classifier = LRClassifier(reduction=reduction,
                              model_type=model_type,
                              class_names=class_type_switcher.get(label),
                              k=k)

    target_class = classifier.get_target_class(image_id)[0]

    print classifier.get_target_class_name(target_class, label)


def do_task_6():
    subject_id = int(raw_input("Enter subject ID: "))
    subject_analyzer = SubjectAnalyzer(_info_path=info_path,
                                       _image_base_path=image_base_path,
                                       _model_type=ModelType.COLOR_MOMENTS)
    print subject_analyzer.get_similar_subjects(subject_id=subject_id)


def do_task_7():
    k = int(raw_input("Enter k: "))
    subject_analyzer = SubjectAnalyzer(_info_path=info_path,
                                       _image_base_path=image_base_path,
                                       _model_type=ModelType.COLOR_MOMENTS)

    reduction = NMFReduction(storage_path=config_provider.get_storage_path(),
                             info_path=config_provider.get_info_path(),
                             model_type=ModelType.COLOR_MOMENTS)

    reduction.df = subject_analyzer.get_sss_matrix()

    data_semantics = reduction.get_data_latent_semantics(k=k)

    term_weight_pairs = reduction.get_latent_data_term_weight_pairs(data_semantics)

    print term_weight_pairs


def do_task_8():
    k = int(raw_input("Enter k: "))
    reduction = NMFReduction(storage_path=config_provider.get_storage_path(),
                             info_path=config_provider.get_info_path(),
                             model_type=ModelType.COLOR_MOMENTS)
    image_ids = [image_factory.get_image_id_from_path(ip) for ip in image_factory.get_image_path_list()]
    reduction.df = class_store.get_class_df_for_image_ids_extrapolated(image_ids)
    data_latent_semantics = reduction.get_data_latent_semantics(k=k)
    term_weight_pairs = reduction.get_latent_data_term_weight_pairs(data_latent_semantics)
    feature_latent_semantics = reduction.get_feature_latent_semantics(k=k)

    data_semantics_plot.plot_semantics(output_path, term_weight_pairs)
    feature_semantics_plot.plot_semantics(output_path, feature_latent_semantics)


def do_load_df():
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    image_id = int(raw_input("Image ID: "))
    print "Selected: " + model_type
    df = do_df_load(model_type, feature_storage)
    features = [f for f in df.loc[image_id, :]]
    print "Feature shape: " + str(len(features))
    print features


def do_get_distances():
    distance_measure = raw_input("Select from [l2, cosine]: ")
    model_type = raw_input("Select from [cm, lbp, hog, sift]: ")
    image_id = int(raw_input("Image ID: "))
    k = int(raw_input("K: "))
    print "Selected: " + model_type
    similar_images = do_distances_get(distance_measure, model_type, image_id, k, feature_storage)
    print similar_images
    plt = do_plot(image_id, similar_images)
    save_plot(plt, image_id,
              {"distance_measure": distance_measure, "model_type": model_type, "query_image_id": image_id, "k": k,
               "similar_images": similar_images})
    plt.show()


extract_item = FunctionItem("Extract Features", do_extract)
task_1_item = FunctionItem("Task 1", do_task_1)
task_2_item = FunctionItem("Task 2", do_task_2)
task_3_item = FunctionItem("Task 3", do_task_3)
task_4_item = FunctionItem("Task 4", do_task_4)
task_5_item = FunctionItem("Task 5", do_task_5)
task_6_item = FunctionItem("Task 6", do_task_6)
task_7_item = FunctionItem("Task 7", do_task_7)
task_8_item = FunctionItem("Task 8", do_task_8)

menu.append_item(task_1_item)
menu.append_item(task_2_item)
menu.append_item(task_3_item)
menu.append_item(task_4_item)
menu.append_item(task_5_item)
menu.append_item(task_6_item)
menu.append_item(task_7_item)
menu.append_item(task_8_item)
menu.append_item(extract_item)

menu.show()
