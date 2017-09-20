import os
import sys
sys.path.append("../../")
from utils import get_root_project_path


SAXO_PROJECT_TEXT_MINING_HOME= get_root_project_path()

processed_files_folder = os.path.join(SAXO_PROJECT_TEXT_MINING_HOME, 'Processed_files')
model_files_folder = os.path.join(SAXO_PROJECT_TEXT_MINING_HOME,
                                  os.path.join('Model_files'))

preprocessed_files_params = os.path.join(processed_files_folder, 'preprocessed.json')
model_files_params = os.path.join(model_files_folder, 'models.json')


tensorboard_log_dir = "logs_tensorboard"