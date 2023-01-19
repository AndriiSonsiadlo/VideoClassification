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

    root_dataset = r"C:\VMShare\datasets\ucf-101\UCF-101"
    root_data = r"C:\VMShare\videoclassification\data"
    root_img_seq_dataset = r"C:\VMShare\videoclassification\data\img_seq_dataset"
    root_models = r"C:\VMShare\videoclassification\data\models"
    root_temp = r"C:\VMShare\videoclassification\data\temp"

    npy_filename = "features.npy"
    video_type = "avi"

