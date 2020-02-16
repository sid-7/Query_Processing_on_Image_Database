import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class ImagePlot:

    def __init__(self, image_id, image_base_path):
        self.image_id = image_id
        self.image_base_path = image_base_path

    def plot_comparison(self, other_image_ids):
        k = len(other_image_ids)

        fig = plt.figure(figsize=(k * 2, 2))
        columns = k + 1
        rows = 1

        image_ids = [self.image_id] + other_image_ids

        for i in range(1, columns * rows + 1):
            image_file_name = self.image_base_path + "Hand_" + str(image_ids[i - 1]).zfill(7) + ".jpg"
            img = mpimg.imread(image_file_name)
            ax = fig.add_subplot(rows, columns, i)
            ax.title.set_text(str(image_ids[i - 1]))
            plt.imshow(img)

        return plt


def test_image_plot():
    image_plot = ImagePlot(2, "/Users/sandeepkunichi/Work/ASU/2019_3/CSE515/DEV/cse515-mwdb-project/data/Hands/")
    ds = [{'s': 0.9989352599000146, 'other_image_id': 3}, {'s': 0.9982293616721234, 'other_image_id': 7},
          {'s': 0.9973345081423015, 'other_image_id': 8}, {'s': 0.9967666983876856, 'other_image_id': 14}]

    image_plot.plot_comparison([node["other_image_id"] for node in ds])
