
from solution3.config import Config
from solution3.data_process.DatasetPreparator import prepare_lists
from solution3.data_process.FeaturesExtractor import FeaturesExtractor
from solution3.data_process.FrameExtractor import FrameExtractor
from solution3.data_process.FileMover import FileMover

def main():
    file_groups = prepare_lists()
    FileMover.move_files(file_groups)

    FrameExtractor.extract_frames()

    extractor = FeaturesExtractor()
    extractor.extract()

if __name__ == '__main__':
    main()
