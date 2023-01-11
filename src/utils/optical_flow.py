import os
import shutil
import random
import warnings

import cv2
import numpy as np
import logging

from skimage.transform import resize
from collections import OrderedDict
from definitions import ROOT_DIR
from src.utils.utils import create_dir

logger = logging.getLogger(__name__)

def stack_optical_flow(frames, mean_sub=False):
    if frames.dtype != np.float32:
        frames = frames.astype(np.float32)
        warnings.warn('Warning! The data type has been changed to np.float32 for graylevel conversion...')
    frame_shape = frames.shape[1:-1]  # e.g. frames.shape is (10, 216, 216, 3)
    num_sequences = frames.shape[0]
    output_shape = frame_shape + (2 * (num_sequences - 1),)  # stacked_optical_flow.shape is (216, 216, 18)
    flows = np.ndarray(shape=output_shape)

    for i in range(num_sequences - 1):
        prev_frame = frames[i]
        next_frame = frames[i + 1]
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = _calc_optical_flow(prev_gray, next_gray)
        flows[:, :, 2 * i:2 * i + 2] = flow

    if mean_sub:
        flows_x = flows[:, :, 0:2 * (num_sequences - 1):2]
        flows_y = flows[:, :, 1:2 * (num_sequences - 1):2]
        mean_x = np.mean(flows_x, axis=2)
        mean_y = np.mean(flows_y, axis=2)
        for i in range(2 * (num_sequences - 1)):
            flows[:, :, i] = flows[:, :, i] - mean_x if i % 2 == 0 else flows[:, :, i] - mean_y

    return flows
def optical_flow_prep(src_dir, dest_dir, mean_sub=True, overwrite=False):
    train_dir = os.path.join(src_dir, 'train')
    test_dir = os.path.join(src_dir, 'test')

    # create dest directory
    create_dir(dest_dir, overwrite=overwrite)
    print(dest_dir, 'created')

    # create directory for training data
    dest_train_dir = os.path.join(dest_dir, 'train')
    if os.path.exists(dest_train_dir):
        print(dest_train_dir, 'already exists')
    else:
        os.mkdir(dest_train_dir)
        print(dest_train_dir, 'created')

    # create directory for testing data
    dest_test_dir = os.path.join(dest_dir, 'test')
    if os.path.exists(dest_test_dir):
        logger.info(f'{dest_test_dir} already exists')
    else:
        os.mkdir(dest_test_dir)
        logger.info(f'{dest_test_dir} created')

    dir_mapping = OrderedDict(
        [(train_dir, dest_train_dir), (test_dir, dest_test_dir)])  # the mapping between source and dest

    logger.info('Start computing optical flows ...')
    for dir, dest_dir in dir_mapping.items():
        logger.info(f'Processing data in {dir}')
        for index, class_name in enumerate(os.listdir(dir)):  # run through every class of video
            class_dir = os.path.join(dir, class_name)
            dest_class_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(dest_class_dir):
                os.mkdir(dest_class_dir)
                # print(dest_class_dir, 'created')
            for filename in os.listdir(class_dir):  # process videos one by one
                file_dir = os.path.join(class_dir, filename)
                frames = np.load(file_dir)
                # note: store the final processed data with type of float16 to save storage
                processed_data = stack_optical_flow(frames, mean_sub).astype(np.float16)
                dest_file_dir = os.path.join(dest_class_dir, filename)
                np.save(dest_file_dir, processed_data)
            # print('No.{} class {} finished, data saved in {}'.format(index, class_name, dest_class_dir))
    logger.info('Finish computing optical flows')
def _calc_optical_flow(prev, next_):
    flow = cv2.calcOpticalFlowFarneback(prev, next_, flow=None, pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                        poly_n=5, poly_sigma=1.2, flags=0)
    return flow


def calc_mean(UCF_dir, img_size):
    frames = []
    logger.info("Calculating RGB mean..")
    for dir_path, dir_names, file_names in os.walk(UCF_dir):
        for filename in file_names:
            path = os.path.join(dir_path, filename)
            if os.path.exists(path):
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    ret, frame = cap.read()
                    # successful read and frame should not be all zeros
                    if ret and frame.any():
                        if frame.shape != (240, 320, 3):
                            frame = resize(frame, (240, 320, 3))
                        frames.append(frame)
                cap.release()
        frames = np.stack(frames)
        mean = frames.mean(axis=0, dtype='int64')
        mean = resize(mean, img_size)
        logger.info(f'RGB mean is calculated over {len(frames)} video frames')
        return mean

def process_frame(frame, img_size, mean=None, normalization=True):
    frame = resize(frame, img_size)
    frame = frame.astype(dtype='float16')
    if mean is not None:
        frame -= mean
    if normalization:
        frame /= 255

    return frame

def process_clip(src_dir, dst_dir, seq_len, img_size, mean=None, normalization=True, continuous_seq=True):
    # logger.debug(f'process-src: {src_dir}')
    # logger.debug(f'process-dst: {dst_dir}')
    all_frames = []
    cap = cv2.VideoCapture(src_dir)
    while cap.isOpened():
        succ, frame = cap.read()
        if not succ:
            break
        # append frame that is not all zeros
        if frame.any():
            all_frames.append(frame)
    # save all frames
    if seq_len is None:
        all_frames = np.stack(all_frames, axis=0)
        dst_dir = os.path.splitext(dst_dir)[0] + '.npy'
        np.save(dst_dir, all_frames)
    else:
        clip_length = len(all_frames)
        if clip_length <= 20:
            logger.info(f'{src_dir}, has not enough frames')
        step_size = int(clip_length / (seq_len + 1))
        frame_sequence = []
        # select random first frame index for continuous sequence
        if continuous_seq:
            start_index = random.randrange(clip_length - seq_len)

        for i in range(seq_len):
            if continuous_seq:
                index = start_index + i
            else:
                index = i * step_size + random.randrange(step_size)
            frame = all_frames[index]
            frame = process_frame(frame, img_size, mean=mean, normalization=normalization)
            frame_sequence.append(frame)
        frame_sequence = np.stack(frame_sequence, axis=0)
        # print(f'DEST[0]: {os.path.splitext(dst_dir)[0]}')
        dst_dir = os.path.splitext(dst_dir)[0] + '.npy'
        # print(f'PROCESS DEST DIR: {dst_dir}')
        np.save(dst_dir, frame_sequence)

    cap.release()

def of_preprocessing(UCF_dir, dest_dir, seq_len, img_size, train_v_test, classes, overwrite=False, normalization=True,
                     mean_subtraction=True, continuous_seq=False):
    '''
    Extract video data to sequence of fixed length, and save it to npy file.
    :param: list_dir
    :param: UCF_dir
    :param: dest_dir
    :param: seq_len
    :param: img_size
    :param: overwrite: whether to overwrite the destination directory
    :param: normalization: normalize to (0,1)
    :return:
    '''

    if os.path.exists(dest_dir):
        if overwrite:
            shutil.rmtree(dest_dir)
        else:
            raise IOError('Destination directory already exists')

    logger.debug(f'dest_dir: {dest_dir}')
    os.mkdir(dest_dir)
    # dest_dir = os.path.join(dest_dir, 'clips_npy')
    # os.mkdir(dest_dir)
    train_dir = os.path.join(dest_dir, 'train')
    test_dir = os.path.join(dest_dir, 'test')
    os.mkdir(train_dir)
    os.mkdir(test_dir)

    if mean_subtraction:
        mean = calc_mean(UCF_dir, img_size).astype(dtype='float16')
        np.save(os.path.join(dest_dir, 'mean.npy'), mean)
    else:
        mean = None

    logger.info('Preprocessing UCF data ...')

    train_list = list()
    test_list = list()
    logger.debug(f'UCF dir: {UCF_dir} ')
    actions = [a for a in os.listdir(UCF_dir) if a in classes]
    # split video file paths into train and test
    for action in actions:
        action_subdir = os.path.join(UCF_dir, action)
        action_clips = os.listdir(action_subdir)
        # print(f'action dir {action_subdir}')
        # print(f'\t{action_clips}')
        k = int(len(action_clips) * train_v_test)
        train_i = random.sample(range(len(action_clips)), k)
        test_i = [j for j in range(len(action_clips)) if j not in train_i]
        train_list.extend([os.path.join(action_subdir, action_clips[i]) for i in train_i])
        test_list.extend([os.path.join(action_subdir, action_clips[i]) for i in test_i])


    logger.debug('Finished splitting train and test')
    #create ucf-preprocessed-OF/vidoes/category


    for clip_list, sub_dir in [(train_list, train_dir), (test_list, test_dir)]:
        logger.debug(f'sub_dir: {sub_dir}')
        logger.debug(f'clip_list: {len(clip_list)}')
        for clip in clip_list:
            clip_name = os.path.basename(clip)
            clip_category = os.path.basename(os.path.dirname(clip))
            # print(f'SUB DIRECTORY: {sub_dir}')
            # print(f'CLIP NAME: {clip_name}')
            # print(f'CLIP CATEGORY: {clip_category}')
            category_dir = os.path.join(sub_dir, clip_category)
            src_dir = os.path.join(UCF_dir, clip)
            dst_dir = os.path.join(category_dir, clip_name)
            # print(f'cat_dir:  {category_dir}')
            # print(f'src_dir: {dst_dir}')
            # print(f'dst_dir: {dst_dir}')
            # print(f'src_dir: {src_dir}')
            if not os.path.exists(category_dir):
                os.mkdir(category_dir)
            process_clip(src_dir, dst_dir, seq_len, img_size, mean=mean, normalization=normalization, continuous_seq=continuous_seq)
    logger.info('Preprocessing done ...')


def get_data_list(video_dir):
    '''
    Input parameters:
    video_dir: directory that stores source train and test data

    Return value:
    test_data/train_data: list of tuples (clip_dir, class index)
    class_index: dictionary of mapping (class_name->class_index)
    '''
    train_dir = os.path.join(video_dir, 'train')
    test_dir = os.path.join(video_dir, 'test')

    testlist = []
    for action in os.listdir(test_dir):
        action_subdir = os.path.join(test_dir, action)
        action_clips = os.listdir(action_subdir)
        testlist.extend([os.path.join(action_subdir, action_clip) for action_clip in action_clips])

    class_index = dict()
    trainlist = []
    for i, action in enumerate(os.listdir(train_dir)):
        action_subdir = os.path.join(train_dir, action)
        action_clips = os.listdir(action_subdir)
        trainlist.extend([os.path.join(action_subdir, action_clip) for action_clip in action_clips])
        class_index[action] = i

    test_data = []
    for filepath in testlist:
        clip_class = os.path.basename(os.path.dirname(filepath))
        test_data.append((filepath, class_index[clip_class]))

    train_data = []
    for filepath in trainlist:
        clip_class = os.path.basename(os.path.dirname(filepath))
        train_data.append((filepath, class_index[clip_class]))

    return train_data, test_data, class_index


def sequence_generator(data_list, batch_size, input_shape, num_classes):
    '''
    Read sequence data of batch_size into memory
    :param data_list: The data generated by get_data_list
    :param batch_size:
    :param input_shape: tuple: the shape of numpy ndarray, e.g. (seq_len, 216, 216, 3) for sequence
                        or (216, 216, 18) for optical flow data
    :param num_classes:
    :return:
    '''
    if isinstance(input_shape, tuple):
        x_shape = (batch_size,) + input_shape
    else:
        raise ValueError('Input shape is neither 1D or 3D')
    y_shape = (batch_size, num_classes)
    index = 0
    while True:
        batch_x = np.ndarray(x_shape)
        batch_y = np.zeros(y_shape)
        for i in range(batch_size):
            step = random.randint(1, len(data_list) - 1)  # approach a random-size step to get the next video sample
            index = (index + step) % len(data_list)
            clip_dir, clip_class = data_list[index]
            batch_y[i, clip_class - 1] = 1
            clip_dir = os.path.splitext(clip_dir)[0] + '.npy'
            # avoid endless loop
            count = 0
            while not os.path.exists(clip_dir):
                count += 1
                if count > 20:
                    raise FileExistsError('Too many file missing')
                index = (index + 1) % len(data_list)
                clip_dir, class_idx = data_list[index]
            clip_data = np.load(clip_dir)
            # print(f'BATCH X: {batch_x.shape[1:]}')
            # print(f'clip shape: {clip_data.shape}')
            if clip_data.shape != batch_x.shape[1:]:
                raise ValueError('The number of time sequence is inconsistent with the video data')
            batch_x[i] = clip_data
        yield batch_x, batch_y


if __name__ == '__main__':
    pass
