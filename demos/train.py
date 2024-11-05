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
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

# data
dataset = LetterFewshotDataset(root=opt.data_root, is_fixed_prompt=True, prob_mask_ref=10)
dataloader = DataLoader(dataset, num_workers=2, batch_size=opt.batch_size, shuffle=True)
print(len(dataset), len(dataset[0]), dataset[0].keys())

# model
model = create_model(opt.model_yaml_path).cpu()

# log
base_logger = pl_loggers.TensorBoardLogger(save_dir=opt.log_dir)
image_logger = ImageLogger(batch_frequency=opt.logger_freq)
ckpt_logger = ModelCheckpoint(dirpath=opt.dst_dir, every_n_train_steps=opt.every_n_train_steps)

# trainer
gpus_nums = len(opt.gpu.split(','))
if gpus_nums > 1:
    trainer = pl.Trainer(gpus=gpus_nums, strategy="ddp", max_epochs=100, logger=base_logger, callbacks=[image_logger, ckpt_logger])
else:
    trainer = pl.Trainer(gpus=gpus_nums, max_epochs=100, logger=base_logger, callbacks=[image_logger, ckpt_logger])
# load model
saved_state_dict = load_state_dict(opt.resume_path, location='cpu')
model.load_state_dict(saved_state_dict)
print('ok...')

# fit
trainer.fit(model, dataloader)
