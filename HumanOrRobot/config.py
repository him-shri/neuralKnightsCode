import sys
import pathlib

working_dir_path = pathlib.Path().absolute()

if sys.platform.startswith('win32'):
    TRAINING_FILES_PATH = str(working_dir_path) + '\\HumanOrRobot\\training_data\\'
    TEST_DATA_PATH = str(working_dir_path) + '\\HumanOrRobot\\test_data\\'
    MODEL_DATA_PATH = str(working_dir_path) + '\\HumanOrRobot\\model\\'
else:
    TRAINING_FILES_PATH = str(working_dir_path) + '/HumanOrRobot/training_data/'
    TEST_DATA_PATH = str(working_dir_path) + '/HumanOrRobot/test_data/'
    MODEL_DATA_PATH = str(working_dir_path) + '/HumanOrRobot/model/'