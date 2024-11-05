import os
import pytorch_lightning as pl
from demos.cfg_py import get_config
from data.Chinese_datasets import LetterFewshotDataset
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from cldm.model import create_model, load_state_dict
from cldm.logger import ImageLogger
from torch.utils.data import DataLoader


opt = get_config()
opt.gpu = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

ckpt_path = '../../ArtGlyphDiffuser/NMCM/10-epoch.ckpt'
test_dir = '../../datasets/font_datasets/SEPARATE/chinese_100/test/01_test_only_gradient'


def get_model(opt, need_load=False, ckpt_path=None):
    model = create_model(opt.model_yaml_path).cpu()
    if need_load:
        print('start load model')
        if ckpt_path is None:
            ckpt_path = opt.resume_path
        saved_state_dict = load_state_dict(ckpt_path, location='cpu')
        model.load_state_dict(saved_state_dict)
        print('load model ok')
    return model


def get_dataloader(opt, shuffle=False):
    # dataset = LetterFewshotDataset(root=opt.data_root, is_fixed_prompt=True, prob_mask_ref=30)
    dataset = LetterFewshotDataset(root=test_dir, is_fixed_prompt=True, prob_mask_ref=0)
    dataloader = DataLoader(dataset, num_workers=2, batch_size=26, shuffle=shuffle)
    print(len(dataset), len(dataset[0]), dataset[0].keys())
    return dataloader


def get_log(opt):
    base_logger = pl_loggers.TensorBoardLogger(save_dir=opt.log_dir)
    image_logger = ImageLogger(batch_frequency=opt.logger_freq)
    ckpt_logger = ModelCheckpoint(dirpath=opt.dst_dir, every_n_train_steps=opt.every_n_train_steps)
    return base_logger, image_logger, ckpt_logger


def get_trainer(opt, ):
    gpus_nums = len(opt.gpu.split(','))
    base_logger, image_logger, ckpt_logger = get_log(opt=opt)
    if gpus_nums > 1:
        trainer = pl.Trainer(gpus=gpus_nums, strategy="ddp", max_epochs=100, logger=base_logger,
                             callbacks=[image_logger, ckpt_logger])
    else:
        trainer = pl.Trainer(gpus=gpus_nums, max_epochs=100, logger=base_logger, callbacks=[image_logger, ckpt_logger])
    return trainer


def train():
    dataloader = get_dataloader(opt)
    model = get_model(opt, need_load=False, ckpt_path=None)
    trainer = get_trainer(opt)
    # fit
    trainer.fit(model, dataloader)


def test():
    dataloader = get_dataloader(opt)
    model = get_model(opt, need_load=True, ckpt_path=ckpt_path)
    saved_dir = './teeeest_dir/'
    model.set_saved_dir(saved_dir=saved_dir)
    # tester
    trainer = pl.Trainer(gpus=1)
    # fit
    trainer.test(model, dataloader)


if __name__ == '__main__':
    test()
