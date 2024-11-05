import os
import clip
import kornia
import torch
import torch.nn as nn
from PIL import Image
from ldm.modules.encoders.modules import AbstractEncoder
from transformers import CLIPTokenizer, CLIPVisionModel, AutoProcessor, CLIPVisionModelWithProjection
from torchvision import transforms


class FrozenOriginCLIPEmbedder(AbstractEncoder):
    def __init__(self, clip_version="ViT-L/14", version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super(FrozenOriginCLIPEmbedder, self).__init__()
        if not os.path.exists(version):
            current_directory = os.path.dirname(__file__)
            version = os.path.join(current_directory, '../../../', version)
        if not os.path.exists(version):
            try:
                saved_pir = os.environ['SAVEDTORCHMODEL']
            except:
                saved_pir = './'
            version = os.path.join(saved_pir, version)

        clip_model, clip_preprocess = clip.load(clip_version)
        self.clip_model = clip_model.to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.device = device
        self.max_length = max_length

    def forward(self, text):
        # tokenizer of stable diffusion
        # batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
        #                                 return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        # tokens = batch_encoding["input_ids"].to(self.device)

        # tokenizer of origin clip model
        tokens = clip.tokenize([desc for desc in text]).to(self.device)
        x = self.clip_model.token_embedding(tokens).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
        return x

    def encode(self, text):
        return self(text)


class FrozenImageToClipEmbedder(nn.Module):
    def __init__(self, version='openai/clip-vit-large-patch14', device='cpu', antialias=False):
        super().__init__()
        if not os.path.exists(version):
            current_directory = os.path.dirname(__file__)
            version = os.path.join(current_directory, '../../../', version)
        if not os.path.exists(version):
            try:
                saved_pir = os.environ['SAVEDTORCHMODEL']
            except:
                saved_pir = './'
            version = os.path.join(saved_pir, version)
        self.clip_vsion_model = CLIPVisionModel.from_pretrained(version).to(device)

        self.device = device
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.freeze()

    def freeze(self):
        self.clip_vsion_model = self.clip_vsion_model.eval()
        for param in self.clip_vsion_model.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic',align_corners=True, antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1, 1]
        x = self.preprocess(x)
        x = self.clip_vsion_model(x).pooler_output
        return x


class FrozenImageToOriginClipEmbedder(nn.Module):
    def __init__(self, version='openai/clip-vit-large-patch14', device='cpu', antialias=False):
        super().__init__()
        if not os.path.exists(version):
            current_directory = os.path.dirname(__file__)
            version = os.path.join(current_directory, '../../../', version)
        if not os.path.exists(version):
            try:
                saved_pir = os.environ['SAVEDTORCHMODEL']
            except:
                saved_pir = './'
            version = os.path.join(saved_pir, version)
        self.clip_vsion_model = CLIPVisionModelWithProjection.from_pretrained(version).to(device)
        self.device = device
        self.antialias = antialias
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)
        self.freeze()

    def freeze(self):
        self.clip_vsion_model = self.clip_vsion_model.eval()
        for param in self.clip_vsion_model.parameters():
            param.requires_grad = False

    def preprocess(self, x):
        # normalize to [0,1]
        x = kornia.geometry.resize(x, (224, 224), interpolation='bicubic',align_corners=True, antialias=self.antialias)
        x = (x + 1.) / 2.
        # renormalize according to clip
        x = kornia.enhance.normalize(x, self.mean, self.std)
        return x

    def forward(self, x):
        # x is assumed to be in range [-1, 1]
        x = self.preprocess(x)
        x = self.clip_vsion_model(x)
        return x.image_embeds


# test FrozenImageToClipEmbedder
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
# img_path = "../../../../../04_Difffusion_model/res/X.png"
# print(os.path.exists(img_path))
#
# clip_image_embedder = FrozenImageToClipEmbedder(device=device, antialias=False)
#
# transform = transforms.Compose([
#     transforms.Resize(size=224),
#     transforms.CenterCrop(size=(224, 224)),
#     transforms.ToTensor(),
#     ])
# image = Image.open(img_path).convert('RGB')
# image = transform(image)
# image = (image - 0.5) * 2
# image = image[None,...]
# image = image.to(device)
#
# image_embed = clip_image_embedder(x=image)
# print(image.shape, image.min(), image.max())
# print(image_embed.shape)


# test FrozenImageToOriginClipEmbedder
# from PIL import Image
# from transformers import AutoProcessor, CLIPVisionModelWithProjection
#
# version = 'openai/clip-vit-large-patch14'
# if not os.path.exists(version):
#     current_directory = os.path.dirname(__file__)
#     version = os.path.join(current_directory, '../../../', version)
# if not os.path.exists(version):
#     try:
#         saved_pir = os.environ['SAVEDTORCHMODEL']
#     except:
#         saved_pir = './'
#     version = os.path.join(saved_pir, version)
#
# model = CLIPVisionModelWithProjection.from_pretrained(version)
# processor = AutoProcessor.from_pretrained(version)
# img_path = "../../../../../04_Difffusion_model/res/X.png"
# image = Image.open(img_path)
# inputs = processor(images=image, return_tensors="pt")
# outputs = model(**inputs)
# image_embeds = outputs.image_embeds
# # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# device = 'cpu'  # 'cuda:2' if torch.cuda.is_available() else 'cpu'
# our_model = FrozenImageToOriginClipEmbedder(device=device, antialias=False)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     ])
# image = Image.open(img_path).convert('RGB')
# image = transform(image)
# image = (image - 0.5) * 2
# image = image[None,...]
# image = image.to(device)
# image_embed_our, _x = our_model(image)
# print(image_embeds.shape, image_embed_our.shape)
# print((image_embeds - image_embed_our).mean())
