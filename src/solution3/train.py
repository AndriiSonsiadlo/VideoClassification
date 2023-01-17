"""
Train our RNN on extracted features or images.
"""
from objects.Model import Model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)


def main():
    """These are the main training settings. Set each before running
    this file."""
    model_name = "lstm"
    batch_size = 32
    nb_epoch = 100
    class_number = 10
    shuffle_classes = False
    video_number_per_class = 10
    shuffle_videos = False
    seq_length = 40
    load_to_memory = False
    test_split = 0.3

    m = Model(model_name=model_name,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              class_number=class_number,
              shuffle_classes=shuffle_classes,
              video_number_per_class=video_number_per_class,
              shuffle_videos=shuffle_videos,
              seq_length=seq_length,
              load_to_memory=load_to_memory,
              test_split=test_split
              )
    m.train()


if __name__ == '__main__':
    fix_gpu()
    main()
