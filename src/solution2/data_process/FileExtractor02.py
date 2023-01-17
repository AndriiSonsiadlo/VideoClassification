"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import csv
import glob
import os
import os.path
from subprocess import call

import ffmpeg
import tqdm
from pandas import DataFrame

from solution2.config import Config
from solution2.data_process.utils import split_path


class FileExtractor:
    @classmethod
    def extract_frames(cls):
        """After we have all of our videos split between train and test, and
        all nested within folders representing their classes, we need to
        make a data file that we can reference when training our RNN(s).
        This will let us keep track of image sequences and other parts
        of the training process.

        We'll first need to extract images from each of the videos. We'll
        need to record the following data in the file:

        [train|test], class, filename, nb frames

        Extracting can be done with ffmpeg:
        `ffmpeg -i video.mpg image-%04d.jpg`
        """
        data_file = []
        folders = [Config.train_folder, Config.test_folder]

        train_folders = glob.glob(os.path.join(folders[0], '*'))
        test_folders = glob.glob(os.path.join(folders[1], '*'))
        pbar = tqdm.tqdm(total=(len(train_folders) + len(test_folders)))

        # in folder train/test
        for folder in folders:
            class_folders = glob.glob(os.path.join(folder, '*'))

            # in class folders
            for class_folder in class_folders:
                pbar.update(1)

                video_folders = glob.glob(os.path.join(class_folder, '*'))

                # in video folder
                for video_folder in video_folders:
                    parts = split_path(video_folder)
                    video_filename = f"{parts[-1]}.{Config.video_type}"
                    video_path = os.path.join(video_folder, video_filename)
                    if not os.path.exists(video_path):
                        print(f"Video {video_filename} is not exists in {video_folder}")
                        continue

                    data = cls.extract_one_file(video_path)
                    data_file.append(data)

            df = DataFrame(data_file)
            df.to_csv(Config.data_file, index=False, header=False)

            print("Extracted and wrote %d video files." % (len(data_file)))

        pbar.close()

    @classmethod
    def extract_one_file(cls, video_path):
        # Get the parts of the file.
        video_parts = cls.get_video_parts(video_path)
        train_or_test, classname, filename_no_ext, filename = video_parts

        # Only extract if we haven't done it yet. Otherwise, just get the info.
        if not cls.check_already_extracted(video_parts):
            # Now extract it.
            src = os.path.join(Config.root_data, train_or_test, classname, filename_no_ext, filename)
            dest = os.path.join(Config.root_data, train_or_test, classname, filename_no_ext, '%04d.jpg')

            command = r"C:\Users\andrii\Downloads\ffmpeg\bin\ffmpeg.exe" + " -i " + src + " " + dest
            os.system(command)
            # call(["ffmpeg", "-i", src, dest])


        # Now get how many frames it is.
        nb_frames = cls.get_nb_frames_for_video(video_parts)

        print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

        data = [train_or_test, classname, filename_no_ext, nb_frames]
        return data

    @classmethod
    def get_nb_frames_for_video(cls, video_parts):
        """Given video parts of an (assumed) already extracted video, return
        the number of frames that were extracted."""
        train_or_test, classname, filename_no_ext, _ = video_parts
        generated_files = glob.glob(os.path.join(Config.root_data, train_or_test, classname, filename_no_ext, '*.jpg'))
        return len(generated_files)

    @classmethod
    def get_video_parts(cls, video_path):
        """Given a full path to a video, return its parts."""
        parts = split_path(video_path)
        filename = parts[-1]
        filename_no_ext = filename.split('.')[0]
        classname = parts[-3]
        train_or_test = parts[-4]

        return train_or_test, classname, filename_no_ext, filename

    @classmethod
    def check_already_extracted(cls, video_parts):
        """Check to see if we created the -0001 frame of this file."""
        train_or_test, classname, filename_no_ext, _ = video_parts
        return bool(
            os.path.exists(os.path.join(Config.root_data, train_or_test, classname, filename_no_ext, '0001.jpg')))


def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    FileExtractor.extract_frames()
    # FileExtractor.extract_one_file(r"C:\VMShare\videoclassification\data\test\BasketballDunk\v_BasketballDunk_g04_c03\v_BasketballDunk_g04_c03.avi")

if __name__ == '__main__':
    main()
