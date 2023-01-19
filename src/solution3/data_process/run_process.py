from src.solution3.data_process.DatasetPreparator import DatasetPreparator
from src.solution3.data_process.FeaturesExtractor import FeaturesExtractor
from src.solution3.data_process.FileMover import FileMover
from src.solution3.data_process.FrameExtractor import FrameExtractor


def main():
    method = "custom"
    class_number = 2
    shuffle_classes = False
    video_number_per_class = 5
    shuffle_videos = False
    seq_length = 40

    videos = DatasetPreparator.prepare_lists(method=method,
                                             class_number=class_number,
                                             shuffle_classes=shuffle_classes,
                                             video_number_per_class=video_number_per_class,
                                             shuffle_videos=shuffle_videos)

    FileMover.move_files(videos)

    FrameExtractor.extract_frames()

    extractor = FeaturesExtractor(seq_length=seq_length)
    extractor.extract()


if __name__ == '__main__':
    main()
