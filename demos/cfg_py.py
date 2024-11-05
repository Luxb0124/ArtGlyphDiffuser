import os
import datetime
from easydict import EasyDict as edict


def get_config():
    current_directory = os.path.dirname(__file__)
    gpi_id = '0'
    model_yaml_path = 'train.yaml'
    model_name = 'NMCM'
    batch_size = 48
    # do not modify it
    base_model_name = 'ArtGlyphDiffuser'
    base_date = '20240703'
    every_n_train_steps = 2000
    logger_freq = 200
    log_root = os.path.join(current_directory, '../../results')
    data_root = os.path.join(current_directory, '../../datasets/font_datasets/SEPARATE/chinese_100/train')
    resume_path = '../../results/PR2024-02550/ArtGlyphDiffuser/NMCM/nmcm_init.ckpt'
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    log_dir = './%s/%s/%s/%s' % (base_model_name, base_date, model_name, current_time)
    log_dir = os.path.join(log_root, log_dir)
    dst_dir = os.path.join(log_dir, 'checkpoints')

    opt = edict()
    opt.gpu = gpi_id
    opt.model_name = model_name
    opt.batch_size = batch_size
    opt.log_root = os.path.join(current_directory, log_root)
    opt.data_root = os.path.join(current_directory, data_root)
    opt.model_yaml_path = os.path.join(current_directory, model_yaml_path)
    opt.resume_path = os.path.join(current_directory, resume_path)
    opt.log_dir = os.path.join(current_directory, log_dir)
    opt.dst_dir = os.path.join(current_directory, dst_dir)
    opt.every_n_train_steps = every_n_train_steps
    opt.logger_freq = logger_freq
    assert os.path.exists(opt.data_root)

    # copy config to running dir
    os.makedirs(opt.dst_dir, exist_ok=True)
    command = 'cp -rf %s %s' %(opt.model_yaml_path, opt.dst_dir)
    print(command)
    os.system(command)
    return opt


# opt = get_config()
# print(opt)
