import glob
import os
import pickle
import shutil

from src.solution3.config import Config


def split_path(txt, seps=("\\", "/", r"\\")):
    default_sep = seps[0]
    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]


def save_model_data_in_pickle(model):
    try:
        with open(os.path.join(model.save_path, "data.pickle"), 'wb') as output:
            pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
            print(fr"Model data was saved in pickle: {model.save_path}\data.pickle")
    except Exception:
        raise (f"Cannot save .pickle {model.save_path}")


def load_pickle_model(path_to_model: str):
    with open(path_to_model, "rb") as input_file:
        model = pickle.load(input_file)
        print(f"Model was loaded: {model.save_path}")
    return model


def cleanup_model_folders():
    """
    Remove each folder in models which not contains 'model' subfolder, where should be stored trained model
    """
    video_folder_paths = glob.glob(f"{Config.root_models}/*")
    for video_folder_path in video_folder_paths:
        if not os.path.exists(os.path.join(video_folder_path, "model")):
            print(f"Removed folder without model in: {video_folder_path}")
            shutil.rmtree(video_folder_path)
