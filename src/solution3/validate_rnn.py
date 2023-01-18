"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger

from objects.Dataset import Dataset
from src.solution3.objects.ModelLoader import ModelLoader


def validate(model_name, seq_length=40, batch_size=32, saved_model=None):

    # Get the data and process it.
    data = Dataset(
        seq_length=seq_length,
    )

    val_generator = data.test_frame_generator(batch_size)

    # Get the model.
    rm = ModelLoader(nb_classes=len(data.get_classes()), model_name=model_name, seq_length=seq_length, saved_model=saved_model)

    # Evaluate!
    results = rm.model.evaluate_generator(
        generator=val_generator)

    print(results)
    print(rm.model.metrics_names)


def main():
    model_name = 'lstm'
    saved_model = r'C:\VMShare\videoclassification\data\checkpoints\lstm-features.046-0.124.hdf5'

    validate(model_name=model_name, saved_model=saved_model)


if __name__ == '__main__':
    main()
