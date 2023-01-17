"""
Validate our RNN. Basically just runs a validation generator on
about the same number of videos as we have in our test set.
"""
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from models import Models
from Dataset import Dataset

def validate(model, seq_length=40, saved_model=None):
    batch_size = 32

    # Get the data and process it.
    data = Dataset(
        seq_length=seq_length,
    )

    val_generator = data.test_frame_generator(batch_size)

    # Get the model.
    rm = Models(len(data.classes), model, seq_length, saved_model)

    # Evaluate!
    results = rm.model.evaluate_generator(
        generator=val_generator,
        val_samples=3200)

    print(results)
    print(rm.model.metrics_names)

def main():
    model = 'lstm'
    saved_model = 'data/checkpoints/lstm-features.026-0.239.hdf5'

    validate(model, saved_model=saved_model)

if __name__ == '__main__':
    main()
