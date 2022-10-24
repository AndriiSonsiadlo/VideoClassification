from src.models import SingleFrameCNN
from src.utils import data_prep

def test_class():
    sfCNN = SingleFrameCNN.SingleFrameCNN

    model = sfCNN.create_model()



