import os
from dataclasses import dataclass


@dataclass
class Config:
    """
    root_dataset: path to dataset videos
    available methods: ["ucflist", "custom"]
        ucflist: use defined ucfList for split train and test files
        random: choose random classes with parameters
        alphabetic: similar as random method, classes is chosen alphabetic

    """

    ################################
    #      Default parameters      #
    ################################

    # root_dataset = "/media/sf_VMShare/datasets/ucf-101/UCF-101"
    root_dataset = r"C:\VMShare\datasets\ucf-101\UCF-101"
    # root_data = r"data"
    root_data = r"C:\VMShare\videoclassification\data"

    data_file = os.path.join(root_data, 'data_file.csv')
    train_folder = os.path.join(root_data, 'train')
    test_folder = os.path.join(root_data, 'test')
    video_type = "avi"

    ################################
    # prepare train and test lists #
    # and move videos to data dir  #
    ################################

    # method for choosing train and test videos
    method = "custom"

    # parameters for "ucf list" method
    version = "01"
    train_list_file = os.path.join(root_data, 'ucfTrainTestlist', 'trainlist' + version + '.txt')
    test_list_file = os.path.join(root_data, 'ucfTrainTestlist', 'testlist' + version + '.txt')

    # parameters for "custom" methods
    shuffle_classes: bool = False
    shuffle_videos: bool = True
    class_number: int | None = 5                # None - all classes
    video_number_per_class: int | None = 5      # None - all videos
    test_split: float = 0.3


    ################################
    #      features extraction     #
    ################################

    # Number of frames to extract features for them
    seq_length_extr = 10
    # Number of classes to extract. Can be 1-101 or None for all.
    class_limit_extr: int | None = None

    ################################
    #      Learning parameters     #
    ################################

    # model can be one of lstm, lrcn, mlp, conv_3d, c3d
    model = 'lstm'
    saved_model = None  # None or weights file
    class_limit_lrn = 10  # int, can be 1-101 or None
    seq_length_lrn = 10
    load_to_memory = False  # preload the sequences into memory
    batch_size = 16
    nb_epoch = 100

    # for lstm, conv_3d, c3d models
    image_shape = (80, 80, 3)





