from src.models.temporal_CNN import temporal_cnn
from src.train_cnn import fit_model
from src.utils.data_prep import generate_data
from definitions import ROOT_DIR
import os
import random

from src.utils.optical_flow import get_data_list

classes = ["Basketball", "PushUps"]


if __name__ == '__main__':
    data_dir = os.path.join(ROOT_DIR, 'data/')
    UCF_dir = os.path.join(ROOT_DIR, 'data/videos/')

    # generate new test/trian data
    # generate_data(data_dir, UCF_dir, classes)

    # model training
    input_shape = (216, 216, 18)
    N_CLASSES = len(classes)
    model = temporal_cnn(input_shape, N_CLASSES, weights_dir='')
    print(model.summary())
    # train CNN using optical flow as input
    weights_dir = ''
    video_dir = os.path.join(ROOT_DIR, data_dir, 'OF_data')
    train_data, test_data, class_index = get_data_list(video_dir)

    print(train_data)
    input_shape = (216, 216, 18)
    fit_model(model, train_data, test_data, weights_dir, input_shape, N_CLASSES, optical_flow=True)