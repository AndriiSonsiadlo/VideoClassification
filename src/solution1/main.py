from solution1.models.model import Model
from solution1.objects.DatasetPreparer import DatasetPreparer
from solution1.objects.VideoLoader import VideoLoader


def main():
    model = Model()
    dp = DatasetPreparer()
    vl = VideoLoader()

    # Dataset preparation
    action_objs = dp.prepare_dataset()
    DatasetPreparer.display_dataset(action_objs)
    train_df, test_df = DatasetPreparer.actions_to_dfs(action_objs)

    # # Feature Extraction
    # feature_extractor = Model().build_feature_extractor()
    #
    # # Label Encoding
    # encoded_label = model.encode_labels(train_df)

    # logging.basicConfig(level=logging.DEBUG)
    # logging.debug("Preparing videos for training")
    train_data, train_labels = model.prepare_all_videos(train_df)
    # logging.debug("Preparing videos for testing")
    test_data, test_labels = model.prepare_all_videos(test_df)

    print(f"Frame features in train set: {train_data[0].shape}")
    print(f"Frame masks in train set: {train_data[1].shape}")

    print(f"train_labels in train set: {train_labels.shape}")
    print(f"test_labels in train set: {test_labels.shape}")

    _, sequence_model = model.run_experiment(train_data, train_labels, test_data, test_labels)

    sequence_model.save("model.h5")

if __name__ == '__main__':
    main()
