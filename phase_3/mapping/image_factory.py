import glob


# Image factory stuff
import re


class ImageFactory:

    def __init__(self, base_path):
        self.base_path = base_path
        self.image_type = ".jpg"

    def get_image_path_list(self):
        _image_paths = [_image_path for _image_path in glob.glob(self.base_path + "/*" + self.image_type)]
        return _image_paths

    @staticmethod
    def get_image_name_from_id(_image_id):
        return "Hand_" + str(_image_id).zfill(7) + ".jpg"

    def get_image_path_from_id(self, _image_id):
        return self.base_path + self.get_image_name_from_id(_image_id)

    @staticmethod
    def get_image_id_from_path(_image_path):
        return re.findall(r'\d+', str(_image_path).split("Hand_")[1].split(".jpg")[0])[0]
