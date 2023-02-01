import tensorflow as tf

from solution1.objects.ConfigReader import ConfigReader


def set_gpu_config():
    hardware_params = ConfigReader()()["hardware"]
    VRAM = hardware_params.getint("VRAM")

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=VRAM)]
            )
        except RuntimeError as e:
            print(e)


