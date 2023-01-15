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

from solution2.config import Config
from solution2.data_process.utils import split_path


class FileExtractor:
    @classmethod
    def extract_files(cls):
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

        # in folder train/test
        for folder in folders:
            class_folders = glob.glob(os.path.join(folder, '*'))

            # in class folders
            for class_folder in class_folders:
                video_folders = glob.glob(os.path.join(class_folder, '*'))

                # in video folder
                for video_folder in video_folders:
                    parts = split_path(video_folder)
                    video_filename = f"{parts[-1]}.{Config.video_type}"
                    video_path = os.path.join(video_folder, video_filename)
                    if not os.path.exists(video_path):
                        print(f"Video {video_filename} is not exists in {video_folder}")
                        continue

                    # Get the parts of the file.
                    video_parts = cls.get_video_parts(video_path)
                    train_or_test, classname, filename_no_ext, filename = video_parts

                    # Only extract if we haven't done it yet. Otherwise, just get the info.
                    if not cls.check_already_extracted(video_parts):
                        # Now extract it.
                        src = os.path.join(Config.root_data, train_or_test, classname, filename_no_ext, filename)
                        dest = os.path.join(Config.root_data, train_or_test, classname, filename_no_ext, '%04d.jpg')

                        # command = "C:\ffmpeg\bin\ffmpeg.exe" + " -i " + src + " " + dest
                        # os.system(command)
                        call(["ffmpeg", "-i", src, dest])

                    # Now get how many frames it is.
                    nb_frames = cls.get_nb_frames_for_video(video_parts)

                    data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                    print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

            with open(Config.data_file, 'w') as fout:
                writer = csv.writer(fout)
                writer.writerows(data_file)

            print("Extracted and wrote %d video files." % (len(data_file)))

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
    FileExtractor.extract_files()


if __name__ == '__main__':
    main()
