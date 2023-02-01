from solution2.config import Config
from solution2.data_process.FileExtractor02 import FileExtractor
from solution2.data_process.FileMover01 import FileMover, prepare_lists
from solution2.extract_features import FeatureExtractor

def main():
    file_groups = prepare_lists()
    FileMover.move_files(file_groups, Config.root_data)
    FileExtractor.extract_frames()

    extractor = FeatureExtractor()
    extractor.extract()

if __name__ == '__main__':
    main()
