import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import AveragePooling2D
from keras.applications import ResNet50
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

