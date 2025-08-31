import os
import logging 
from from_root import from_root
from datetime import datetime

log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = 'logs'

log_path = os.path.join(from_root(), log_dir)
os.makedirs(log_path, exist_ok= True)

log_file_path = os.path.join(log_path, log_file)


logging.basicConfig(
    filename = log_file_path, 
    format = "[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO   
)