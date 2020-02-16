import json


class ConfigProvider:

    def __init__(self, config_path):
        with open(config_path) as json_file:
            self.config = json.load(json_file)

    def get_image_base_path_train(self):
        return self.config.get("image_base_path_train")

    def get_image_base_path_test(self):
        return self.config.get("image_base_path_test")

    def get_storage_path_train(self):
        return self.config.get("storage_path_train")

    def get_storage_path_test(self):
        return self.config.get("storage_path_test")

    def get_output_path(self):
        return self.config.get("output_path")

    def get_info_path(self):
        return self.config.get("info_path")
