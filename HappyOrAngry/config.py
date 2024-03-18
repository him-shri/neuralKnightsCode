import sys
import pathlib

working_dir_path = pathlib.Path().absolute()

if sys.platform.startswith('win32'):
    TRAINING_FILES_PATH = str(working_dir_path) + '\\HappyOrAngry\\training_data\\'
    TEST_DATA_PATH = str(working_dir_path) + '\\HappyOrAngry\\test_data\\'
    MODEL_DATA_PATH = str(working_dir_path) + '\\HappyOrAngry\\model\\'
else:
    TRAINING_FILES_PATH = str(working_dir_path) + '/HappyOrAngry/training_data/'
    TEST_DATA_PATH = str(working_dir_path) + '/HappyOrAngry/test_data/'
    MODEL_DATA_PATH = str(working_dir_path) + '/HappyOrAngry/model/'