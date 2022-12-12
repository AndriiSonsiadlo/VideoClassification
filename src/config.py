from dataclasses import dataclass
from os.path import dirname, abspath


@dataclass
class DatasetConfig:
    dataset_path = "data/videos/"
    video_file_type = "avi"
    class_number = 2
    video_class_number = 2
    test_dataset_in_percent = 0.25

    # person
    folder_persons_data = "person_data"
    folder_temp = "temp"
    path_temp = f"{folder_persons_data}//{folder_temp}"
    folder_person_photo = "photos"

    file_person_json = "person_data.json"
    file_person_list_pkl = "person_list.pkl"
    dir_person_data = dirname(
        dirname(abspath(__file__))) + f"\\{folder_persons_data}\\"  # 'Kivy/person_data' directory path


@dataclass
class LearningConfig:
    save_model_path = "data/models/"
    max_video_frames = 30
    img_size_w = 224
    img_size_h = 224
    batch_size = 64
    epochs = 200
    max_seq_length = 50
    num_features = 2048

    allowed_extensions = {'png', 'jpg', 'jpeg', 'JPG', 'JPEG', 'PNG'}

    # model
    file_model_json = 'model_data.json'
    filename_model_list_pkl = 'model_list.pkl'
    filename_svm_model = 'svm_model.clf'
    filename_knn_model = 'knn_model.clf'
    folder_models_data = "model_data"
    dir_model_data = dirname(dirname(abspath(__file__))) + f"\\{folder_models_data}\\"


    algorithm_knn = "KNN Classification"
    algorithm_svm = "SVM Classification"
    algorithms_values = [algorithm_knn, algorithm_svm]
    uniform = "uniform"
    distance = "distance"
    weights_values = [distance, uniform]  # KNN
    auto = "auto"
    scale = "scale"
    gamma_values = [auto, scale]  # SVN

    threshold_default = 0.5
    # default is a 5 frames to identificate person
    default_count_frame = 5
    default_count_photos = 1


class StatisticsConfig:
    path_file_stats = "src/statistics/basic_data.csv"
    path_plt_facescreen_stats = "src/statistics/result.png"
    file_plot = 'plot.png'


@dataclass
class HardwareConfig:
    is_GPU = True
    VRAM = 2048



@dataclass
class JsonKeyConfig:
    information = "information"
    model_name = "Model name"
    author = "Author"
    comment = "Comment"
    p_date = "Date"
    p_time = "Time"
    learning_time = "Learning time [in sec]"

    # Recognition algorithm
    n_neighbor = "n_neighbor"
    weights = "Weights"
    gamma = "Gamma"
    algorithm = "Algorithm"
    accuracy = "Accuracy"
    threshold = "Threshold distance"

    count_train_dataset = "Count train dataset"
    count_test_dataset = "Count test dataset"
    train_dataset = "Train dataset"
    test_dataset = "Test dataset"

    p_is_wanted = "Is Wanted"
    p_name = "Full name"
    p_age = "Age"
    p_gender = "Gender"
    p_nationality = "Nationality"
    p_details = "Details"
    p_contact_phone = "Phone for contact"
    p_photo_paths = "Photo paths"
    p_count_photo = "Count photo"


@dataclass
class PostgresConfig:
    host = "localhost"
    user = "postgres"
    password = "postgres"
    db = "videoclassification"


@dataclass
class ProjectInfoConfig:
    project = "Video Classification"
    authors = "Andrii Sonsiadlo, Krzysztof Ragan, Stanislaw Kolakowski"
    description = "It's VideoClassification project"


@dataclass
class CustomizationConfig:
    header_text_color = (0 / 255, 102 / 255, 178 / 255, 1)
    normal_text_color = (0.2, 0.2, 0.2, 1)

    text_unnamed = "Unnamed"
    text_unknown = "Unknown"
    text_learning = "Learning..."
    text_train_model = "Train model"
    text_completed = "Completed"
    no_elements_text = "No elements"

    camera_off_path_image = "src/assets/Images/camera_off_2.png"
    default_user_image = "src/assets/Images/default-user.png"

    start_webcam_text = "Turn on"
    stop_webcam_text = "Turn off"

    port_0 = "Port 0"
    port_1 = "Port 1"
    port_2 = "Port 2"
    port_3 = "Port 3"
    port_4 = "Port 4"
    camera_values = [port_0, port_1, port_2, port_3, port_4]










