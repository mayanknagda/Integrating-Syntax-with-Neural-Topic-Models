import os
import pickle


def dump_pkl(root_path, data, name):
    """ This function is used to dump any data into a pickle file."""
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    pickle.dump(data, open(os.path.join(root_path, name + '.pkl'), 'wb'))


def load_pkl(root_path, name):
    """ This function is used to load any data from a pickle file."""
    return pickle.load(open(os.path.join(root_path, name + '.pkl'), 'rb'))
