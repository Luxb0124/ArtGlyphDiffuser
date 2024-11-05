import os
import datetime
from easydict import EasyDict as edict


def get_config():
    current_directory = os.path.dirname(__file__)
    gpi_id = '2'
    model_yaml_path = '../agd_cfg/v3.7-vcdm_v15.yaml'
    model_name = 'V2SCP_multi-level'
    batch_size = 96
    # do not modify it
    base_model_name = 'v2csp_context'
    base_date = '20240315'
    every_n_train_steps = 2000
    logger_freq = 200
    log_root = '../../../../../results'
    data_root = '../../../../../datasets/font_datasets/SEPARATE/Capitals_colorGrad64/train'
    resume_path = '../../checkpoints/v2scp_multi_level_sd15_ini.ckpt'
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
