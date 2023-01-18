"""
Train our RNN on extracted features or images.
"""
import tensorflow as tf
from src.solution3.objects.ModelData import ModelData, save_pickle_model



def fix_gpu():
    # from tensorflow.compat.v1 import ConfigProto
    # from tensorflow.compat.v1 import InteractiveSession
    # config = ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = InteractiveSession(config=config)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


def main():
    """These are the main training settings. Set each before running
    this file."""
    model_name = "lstm"
    batch_size = 64
    nb_epoch = 100
    class_number = 10
    shuffle_classes = True
    video_number_per_class = 15
    shuffle_videos = True
    seq_length = 40
    load_to_memory = False
    test_split = 0.3

    m = ModelData(model_name=model_name,
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
    history = m.train()
    save_pickle_model(m)
    m.save_to_json()
    m.train_plot(history.history)

if __name__ == '__main__':
    fix_gpu()
    main()
