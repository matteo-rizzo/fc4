import os
import pickle
import sys

from utils import print_angular_errors


def load_errors(model_name: str):
    if model_name.endswith('.pkl'):
        return pickle.load(open(model_name))
    path_to_model = os.path.join("../models", "fc4", model_name)
    file_name = list(sorted(filter(lambda x: x.startswith("error"), os.listdir(path_to_model))))[-1]
    return pickle.load(open(os.path.join(path_to_model, file_name)))


def main(models: list):
    print_angular_errors([load_errors(model) for model in models])


if __name__ == '__main__':
    main(sys.argv[1:])
