from skimage import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms, datasets
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


def load_pretrained_AlexNet(model, progress=True):
    __all_ = ["AlexNet", "alexnet", "Alexnet"]
    model_url = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }

    print("loading pre-trained AlexNet...")
    state_dict = load_state_dict_from_url(model_url['alexnet'], progress=progress)
    model_dict = model.state_dict()

    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}

    model_dict.update(state_dict)
    model.load_state_dict(state_dict)
    print("loaded model correctly...")


def save_log(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        print('[INFO] Object saved to {}'.format(path))


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print("checkpoint saved in {}".format(path))


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print("pre-trained model loaded from {}".format(path))
