import logging
import math

import numpy as np
from tensorflow import keras
from tqdm import tqdm

from objects.ConfigReader import ConfigReader
from objects.VideoLoader import VideoLoader


class Model:

    def __init__(self):
        self.label_processor = None

        learning_params = ConfigReader()()["learning"]
        self.num_features = learning_params.getint("num_features")
        self.max_seq_length = learning_params.getint("max_seq_length")
        self.epochs = learning_params.getint("epochs")
        self.batch_size = learning_params.getint("batch_size")
        self.max_video_frames = learning_params.getint("max_video_frames")
        img_size_w = learning_params.getint("img_size_w")
        img_size_h = learning_params.getint("img_size_h")
        self.img_resize = (img_size_h, img_size_w)

    def read_config(self):
        pass

    def build_feature_extractor(self):
        feature_extractor = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=(*self.img_resize, 3),
        )
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input((*self.img_resize, 3))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")

    def encode_labels(self, df):
        self.label_processor = keras.layers.StringLookup(num_oov_indices=0, vocabulary=np.unique(df["label"]))
        labels = df["label"].values
        labels = self.label_processor(labels[..., None]).numpy()
        return labels

    def prepare_all_videos(self, df):
        num_samples = len(df)
        video_paths = df["video_path"].values.tolist()

        labels = self.encode_labels(df)

        # `frame_masks` and `frame_features` are what we will feed to our sequence model.
        # `frame_masks` will contain a bunch of booleans denoting if a timestep is
        # masked with padding or not.
        frame_masks = np.zeros(shape=(num_samples, self.max_seq_length), dtype="bool")  # 145,20
        frame_features = np.zeros(shape=(num_samples, self.max_seq_length, self.num_features),
                                  dtype="float32")  # 145,20,2048

        build_extractor = self.build_feature_extractor()

        # For each video.
        for idx, path in enumerate(tqdm(video_paths, colour="green")):
            # Gather all its frames and add a batch dimension.
            frames = VideoLoader().load_video(path, img_resize=self.img_resize)

            frame_batches = []
            batch = []
            for frame in frames:
                batch.append(frame)
                if len(batch) == self.batch_size or (frame == frames[-1]).all():
                    frame_batches.append(np.array(batch))
                    batch.clear()

            # frames = np.array_split(frames, math.ceil(len(frames) / self.batch_size))
            # frames = np.array_split(frames, self.batch_size)
            # frames = frames[None, ...]

            # Initialize placeholders to store the masks and features of the current video.
            temp_frame_mask = np.zeros(shape=(1, self.max_seq_length), dtype="bool")
            temp_frame_features = np.zeros(
                shape=(1, self.max_seq_length, self.num_features), dtype="float32"
            )

            # TODO fix below code in loop, used only index 0, must be < i >
            # Extract features from the frames of the current video.
            for i, batch in enumerate(frame_batches):
                video_length = batch.shape[0]
                length = min(self.max_seq_length, video_length)
                for j in range(length):
                    temp_frame_features[0, j, :] = build_extractor.predict(batch[None, j, :], verbose=False)
                temp_frame_mask[0, :length] = 1  # 1 = not masked, 0 = masked

            frame_features[idx] = temp_frame_features.squeeze()
            frame_masks[idx] = temp_frame_mask.squeeze()

        return (frame_features, frame_masks), labels

    def get_sequence_model(self):
        class_vocab = self.label_processor.get_vocabulary()

        frame_features_input = keras.Input((self.max_seq_length, self.num_features))
        mask_input = keras.Input((self.max_seq_length,), dtype="bool")

        # Refer to the following tutorial to understand the significance of using `mask`:
        # https://keras.io/api/layers/recurrent_layers/gru/
        x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
        x = keras.layers.GRU(8)(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.Dense(8, activation="relu")(x)
        output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

        rnn_model = keras.Model([frame_features_input, mask_input], output)

        rnn_model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        return rnn_model

    def run_experiment(self, train_data, train_labels, test_data, test_labels):
        filepath = "./data"
        checkpoint = keras.callbacks.ModelCheckpoint(
            filepath, save_weights_only=True, save_best_only=True, verbose=1
        )

        seq_model = self.get_sequence_model()
        history = seq_model.fit(
            [train_data[0], train_data[1]],
            train_labels,
            validation_split=0.3,
            epochs=self.epochs,
            callbacks=[checkpoint],
        )

        seq_model.load_weights(filepath)
        _, accuracy = seq_model.evaluate([test_data[0], test_data[1]], test_labels)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

        # probabilities = seq_model.predict([frame_features, frame_mask])[0]

        return history, seq_model

    def save_model(self):
        pass

    def read_model(self):
        pass
