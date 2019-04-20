
import json
import os.path

import requests
from hbconfig import Config


def send_message_to_slack(config_name):
    project_name = os.path.basename(os.path.abspath("."))

    data = {
        "text": f"The learning is finished with *{project_name}* Project using `{config_name}` config."
    }

    webhook_url = Config.slack.webhook_url
    if webhook_url == "":
        print(data["text"])
    else:
        requests.post(Config.slack.webhook_url, data=json.dumps(data))

import matplotlib.pyplot as plt

label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
    ]


def plot_images(images, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    fig, axes = plt.subplots(3, 3)
    
    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')
        
        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
                )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.show()


def saving_config(path):
    with open(path, "w+") as text_file:
        text_file.write(f"Config: {Config}")
        if Config.get("description", None):
            text_file.write("Config: {}".format(Config))
            text_file.write("Config Description")
            for key, value in Config.description.items():
                text_file.write(f" - {key}: {value}")
