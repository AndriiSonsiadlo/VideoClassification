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

    root_dataset = r"D:\MAGISTERSKIE\Uczenie_glebokie\Action_Classification_CNN\videoclassification\data\videos"
    root_data = r"D:\MAGISTERSKIE\Uczenie_glebokie\Action_Classification_CNN\videoclassification\data"
    root_img_seq_dataset = r"D:\MAGISTERSKIE\Uczenie_glebokie\Action_Classification_CNN\videoclassification\data\img_seq_dataset"
    root_models = r"D:\MAGISTERSKIE\Uczenie_glebokie\Action_Classification_CNN\videoclassification\data\models"
    root_temp = r"D:\MAGISTERSKIE\Uczenie_glebokie\Action_Classification_CNN\videoclassification\data\temp"

    npy_filename = "features.npy"
    video_type = "avi"

