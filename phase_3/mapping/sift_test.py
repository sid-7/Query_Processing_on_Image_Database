import json

import cv2
from image_factory import ImageFactory
from tqdm import tqdm

from phase_1.sandeepkunichi.plot.image_plot import ImagePlot


def extract_distance(json):
    try:
        return float(json["s"])
    except KeyError:
        return 0


with open('/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/store/sift.json') as json_file:
    data = json.load(json_file)

print data

'''
bf = cv2.BFMatcher()
image_factory = ImageFactory("/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/Hands/")
train_path = "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/Hands/Hand_0000002.jpg"
sift = cv2.xfeatures2d.SIFT_create()
train_img_rgb = cv2.imread(train_path)
train_img = cv2.cvtColor(train_img_rgb, cv2.COLOR_BGR2GRAY)
kp_train, des_train = sift.detectAndCompute(train_img, None)

match_scores = []
for path in tqdm(image_factory.get_image_path_list()):
    query_img_rgb = cv2.imread(path)
    query_img = cv2.cvtColor(query_img_rgb, cv2.COLOR_BGR2GRAY)
    kp_query, des_query = sift.detectAndCompute(query_img, None)
    s = len([[m] for m, n in bf.knnMatch(des_train, des_query, k=2) if m.distance < 0.8 * n.distance])
    match_scores.append({"s": s, "other_image_id": image_factory.get_image_id_from_path(path)})

match_scores.sort(key=extract_distance, reverse=True)

image_plot = ImagePlot(2, "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/Hands/")

image_plot.plot_comparison([node["other_image_id"] for node in match_scores][:4])
'''