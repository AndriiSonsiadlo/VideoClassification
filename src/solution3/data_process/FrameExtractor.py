"""
After moving all the files using the 1_ file, we run this one to extract
the images from the videos and also create a data file we can use
for training and testing later.
"""
import glob
import os
import os.path

import tqdm

from src.solution3.config import Config
from src.solution3.utils import split_path


class FrameExtractor:
    cfg = Config()

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
        dataset_dir = cls.cfg.root_img_seq_dataset
        class_folders = glob.glob(os.path.join(dataset_dir, '*'))

        pbar = tqdm.tqdm(total=len(class_folders))

        # in class folders
        for class_folder in class_folders:
            pbar.update(1)

            video_folders = glob.glob(os.path.join(class_folder, '*'))

            # in video folder
            for video_folder in video_folders:
                parts = split_path(video_folder)
                video_filename = f"{parts[-1]}.{cls.cfg.video_type}"
                video_path = os.path.join(video_folder, video_filename)

                if not os.path.exists(video_path):
                    print(f"Video {video_filename} is not exists in {video_folder}")
                    continue

                cls.extract_frames_for_one_video(video_path)

        print("Extracted and wrote %d video files." % (len(data_file)))
        pbar.close()

    @classmethod
    def extract_frames_for_one_video(cls, video_path):
        # Get the parts of the file.
        video_parts = cls.get_video_parts(video_path)
        classname, filename_no_ext, filename = video_parts

        # Only extract if we haven't done it yet. Otherwise, just get the info.
        if not cls.check_already_extracted(classname=classname, filename_no_ext=filename_no_ext):
            # Now extract it.
            src = os.path.join(cls.cfg.root_img_seq_dataset, classname, filename_no_ext, filename)
            dest = os.path.join(cls.cfg.root_img_seq_dataset, classname, filename_no_ext, '%04d.jpg')

            command = rf"C:\Users\andrii\Downloads\ffmpeg\bin\ffmpeg.exe -i {src} {dest}"
            os.system(command)
            # call(["ffmpeg", "-i", src, dest]) # Linux

        # Now get how many frames it is.
        video_folder_path = os.path.join(cls.cfg.root_img_seq_dataset, classname, filename_no_ext)
        nb_frames = cls.get_nb_frames_for_video(video_folder_path)

        print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    def extract_frames_for_one_video_prediction(cls, video_path, id):
        # Get the parts of the file.
        video_parts = cls.get_video_parts(video_path)
        classname, filename_no_ext, filename = video_parts

        # Only extract if we haven't done it yet. Otherwise, just get the info.
        if not cls.check_already_extracted(classname=classname, filename_no_ext=filename_no_ext):
            # Now extract it.
            src = os.path.join(cls.cfg.root_temp, str(id), filename)
            dest = os.path.join(cls.cfg.root_temp, str(id),  '%04d.jpg')

            command = rf"C:\Users\andrii\Downloads\ffmpeg\bin\ffmpeg.exe -i {src} {dest}"
            os.system(command)
            # call(["ffmpeg", "-i", src, dest]) # Linux

        # Now get how many frames it is.
        nb_frames = cls.get_nb_frames_for_video(os.path.join(cls.cfg.root_temp, str(id)))

        print("Generated %d frames for temp/%s: video %s" % (nb_frames, str(id), filename))


    @classmethod
    def get_nb_frames_for_video(cls, video_folder_path):
        """Given video parts of an (assumed) already extracted video, return
        the number of frames that were extracted."""
        generated_files = glob.glob(os.path.join(video_folder_path, '*.jpg'))
        return len(generated_files)

    @classmethod
    def get_video_parts(cls, video_path):
        """Given a full path to a video, return its parts."""
        parts = split_path(video_path)
        filename = parts[-1]
        filename_no_ext = filename.split('.')[0]
        classname = parts[-3]

        return classname, filename_no_ext, filename

    @classmethod
    def check_already_extracted(cls, classname, filename_no_ext):
        """Check to see if we created the -0001 frame of this file."""
        return bool(
            os.path.exists(os.path.join(cls.cfg.root_img_seq_dataset, classname, filename_no_ext, '0001.jpg'))
        )


def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:

    [train|test], class, filename, nb frames
    """
    FrameExtractor.extract_frames()
    # FileExtractor.extract_one_file(r"C:\VMShare\videoclassification\data\test\BasketballDunk\v_BasketballDunk_g04_c03\v_BasketballDunk_g04_c03.avi")


if __name__ == '__main__':
    main()
