"""
Train our RNN on extracted features or images.
"""
import glob
import os.path
import shutil

import tensorflow as tf

from src.solution3.config import Config
from src.solution3.objects.ModelData import ModelData
from src.solution3.utils import save_model_data_in_pickle, cleanup_model_folders


def fix_gpu():
    """ I don't know what this function do XDD """
    # from tensorflow.compat.v1 import ConfigProto
    # from tensorflow.compat.v1 import InteractiveSession
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


def run_training():
    """
    These are the main training settings. Set each before running this function

    Variables:
        model_name: available values: ["lstm", "rnn", "gru"]

        run_num: used in model path, e.g.: data/models/{run_num}_models_id
        batch_size: video num for one epoch
        nb_epoch: epoch number
        included_classes: list of classes will be used in training model, if length of list is less than number
            of classes, list will be filled of random classes
        class_number: number of classes
        shuffle_classes: (True) classes are shuffled while preparing dataset for a training, then classes are randomly chosen
            or (False) alphabetically chosen
        video_number_per_class: number of video for one class
        shuffle_videos: (True) videos are shuffled or (False) alphabetically
        seq_length: number of frame features for one video
        load_to_memory: (True) load all features to RAM or (False) created generator
        test_split: percent of test dataset
    """
    run_num = "1"

    model_name = "lstm"
    batch_size = 32
    nb_epoch = 40
    # WARN! Should be tuple, else these classes won't be read
    incl_classes = tuple()
    class_number = 10
    shuffle_classes = True
    video_number_per_class = 15
    shuffle_videos = True
    seq_length = 40
    load_to_memory = False
    test_split = 0.3

    model_data = ModelData(model_name=model_name,
                           batch_size=batch_size,
                           nb_epoch=nb_epoch,
                           incl_classes=incl_classes,
                           class_number=class_number,
                           shuffle_classes=shuffle_classes,
                           video_number_per_class=video_number_per_class,
                           shuffle_videos=shuffle_videos,
                           seq_length=seq_length,
                           load_to_memory=load_to_memory,
                           test_split=test_split,
                           run_num=run_num
                           )

    # while True:
    #     try:
    #         history = model.train()
    #     except:
    #         cleanup_model_folders()
    #     else:
    #         break

    model = model_data.train()
    save_model_data_in_pickle(model_data)
    model_data.train_plot(model.history)
    model_data.save_to_json()


def run_training_for_setups():
    """
    Function used to train several models. Parameters defined in the variable 'setups'

    Variables:
        run_num: used in model path, e.g.: data/models/{run_num}_models_id
        batch_size: video num for one epoch
        nb_epoch: epoch number
        shuffle_classes: (True) classes are shuffled while preparing dataset for a training, then classes are randomly chosen
            or (False) alphabetically chosen
        video_number_per_class: number of video for one class
        shuffle_videos: (True) videos are shuffled or (False) alphabetically
        seq_length: number of frame features for one video
        load_to_memory: (True) load all features to RAM or (False) created generator
        test_split: percent of test dataset
        setups:
            model: ["lstm", "rnn", "gru"]
            included_classes: list of classes will be used in training model, if length of list is less than number
                of classes, list will be filled of random classes
            class_number: number of classes
    """
    setups = [

        (
            "lstm",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "BalanceBeam",
                "Biking",
                "FrontCrawl",
                "JumpingJack",
                "PushUps",
                "ShavingBeard",
                "TrampolineJumping",
                "UnevenBars"
            ),
            10
        ),
        (
            "lstm",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "BabyCrawling",
                "BalanceBeam",
                "BandMarching",
                "Biking",
                "BlowDryHair",
                "BlowingCandles",
                "BoxingSpeedBag",
                "FrontCrawl",
                "JumpingJack",
                "Kayaking",
                "Lunges",
                "Nunchucks",
                "PullUps",
                "PushUps",
                "ShavingBeard",
                "StillRings",
                "TrampolineJumping",
                "UnevenBars"
            ),
            20
        ),
        (
            "lstm",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "Archery",
                "BabyCrawling",
                "BalanceBeam",
                "BandMarching",
                "Biking",
                "Billiards",
                "BlowDryHair",
                "BlowingCandles",
                "BoxingPunchingBag",
                "BoxingSpeedBag",
                "CricketBowling",
                "Fencing",
                "FrontCrawl",
                "GolfSwing",
                "IceDancing",
                "JumpingJack",
                "Kayaking",
                "Lunges",
                "Nunchucks",
                "PullUps",
                "PushUps",
                "ShavingBeard",
                "StillRings",
                "TableTennisShot",
                "TrampolineJumping",
                "Typing",
                "UnevenBars",
                "YoYo"
            ),
            30,
        ),
        (
            "gru",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "BalanceBeam",
                "Biking",
                "FrontCrawl",
                "JumpingJack",
                "PushUps",
                "ShavingBeard",
                "TrampolineJumping",
                "UnevenBars"
            ),
            10
        ),
        (
            "gru",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "BabyCrawling",
                "BalanceBeam",
                "BandMarching",
                "Biking",
                "BlowDryHair",
                "BlowingCandles",
                "BoxingSpeedBag",
                "FrontCrawl",
                "JumpingJack",
                "Kayaking",
                "Lunges",
                "Nunchucks",
                "PullUps",
                "PushUps",
                "ShavingBeard",
                "StillRings",
                "TrampolineJumping",
                "UnevenBars"
            ),
            20
        ),
        (
            "gru",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "Archery",
                "BabyCrawling",
                "BalanceBeam",
                "BandMarching",
                "Biking",
                "Billiards",
                "BlowDryHair",
                "BlowingCandles",
                "BoxingPunchingBag",
                "BoxingSpeedBag",
                "CricketBowling",
                "Fencing",
                "FrontCrawl",
                "GolfSwing",
                "IceDancing",
                "JumpingJack",
                "Kayaking",
                "Lunges",
                "Nunchucks",
                "PullUps",
                "PushUps",
                "ShavingBeard",
                "StillRings",
                "TableTennisShot",
                "TrampolineJumping",
                "Typing",
                "UnevenBars",
                "YoYo"
            ),
            30,
        ),

        (
            "rnn",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "BalanceBeam",
                "Biking",
                "FrontCrawl",
                "JumpingJack",
                "PushUps",
                "ShavingBeard",
                "TrampolineJumping",
                "UnevenBars"
            ),
            10
        ),
        (
            "rnn",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "BabyCrawling",
                "BalanceBeam",
                "BandMarching",
                "Biking",
                "BlowDryHair",
                "BlowingCandles",
                "BoxingSpeedBag",
                "FrontCrawl",
                "JumpingJack",
                "Kayaking",
                "Lunges",
                "Nunchucks",
                "PullUps",
                "PushUps",
                "ShavingBeard",
                "StillRings",
                "TrampolineJumping",
                "UnevenBars"
            ),
            20
        ),
        (
            "rnn",
            # WARN! Should be tuple, else these classes won't be read
            (
                "ApplyEyeMakeup",
                "ApplyLipstick",
                "Archery",
                "BabyCrawling",
                "BalanceBeam",
                "BandMarching",
                "Biking",
                "Billiards",
                "BlowDryHair",
                "BlowingCandles",
                "BoxingPunchingBag",
                "BoxingSpeedBag",
                "CricketBowling",
                "Fencing",
                "FrontCrawl",
                "GolfSwing",
                "IceDancing",
                "JumpingJack",
                "Kayaking",
                "Lunges",
                "Nunchucks",
                "PullUps",
                "PushUps",
                "ShavingBeard",
                "StillRings",
                "TableTennisShot",
                "TrampolineJumping",
                "Typing",
                "UnevenBars",
                "YoYo"
            ),
            30,
        ),
    ]

    run_num = "1"
    batch_size = 32
    nb_epoch = 100
    shuffle_classes = True
    video_number_per_class = 15
    shuffle_videos = True
    seq_length = 40
    load_to_memory = False
    test_split = 0.3

    for i, (model_name, incl_classes, class_number) in enumerate(setups[7:]):
        print("*" * 20, f"Processing setup {i}", "*" * 20)

        print(model_name)
        print(incl_classes)
        print(class_number)

        m = ModelData(
            model_name=model_name,
            batch_size=batch_size,
            nb_epoch=nb_epoch,
            incl_classes=incl_classes,
            class_number=class_number,
            shuffle_classes=shuffle_classes,
            video_number_per_class=video_number_per_class,
            shuffle_videos=shuffle_videos,
            seq_length=seq_length,
            load_to_memory=load_to_memory,
            test_split=test_split,
            run_num=run_num
        )
        history = m.train()
        save_model_data_in_pickle(m)
        m.train_plot(history.history)
        m.save_to_json()


if __name__ == '__main__':
    fix_gpu()
    run_training()

    # run_training_for_setups()
    # cleanup_model_folders()
