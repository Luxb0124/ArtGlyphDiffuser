import os
import glob
import random
import numpy as np
import torch.utils.data as data

from PIL import Image
from torch.utils.data import DataLoader


class LetterFewshotDataset(data.Dataset):
    def __init__(self, root=None, few_num=3, nencode=1, test_nums=None, dim=(64, 64), is_fixed_prompt=True, prob_mask_ref=0):
        self.root = root
        self.data = []
        self.dim = dim
        self.few_num = few_num
        self.nencode = nencode
        assert (self.nencode <= self.few_num)

        self.styles = glob.glob('%s/*' % (self.root))
        self.styles = self.styles[:test_nums]
        print('the number of styles', len(self.styles))
        if not is_fixed_prompt:
            self.prompts = self.get_prompts()
        else:
            self.prompts = None
        self.alphabets = self.get_alphabets(self.styles[0])
        self.prob_mask_ref = prob_mask_ref
        print('the number of alphabets', len(self.alphabets), ' the prob of mask style,', self.prob_mask_ref)

    def get_alphabets(self, style):
        all_png_paths = glob.glob('%s/*.png' %(style))
        alphabets = [os.path.basename(png_path)[0] for png_path in all_png_paths]
        return alphabets

    def __len__(self):
        return len(self.styles) * len(self.alphabets)

    def get_prompts(self):
        current_directory = os.path.dirname(__file__)
        prompts_path = os.path.join(current_directory, 'chinese_prompts.txt')
        with open(prompts_path, 'r') as f:
            prompts = f.readlines()
        return prompts

    def __getitem__(self, idx):
        style = self.styles[idx // len(self.alphabets)]
        # random sample target character
        random.shuffle(self.alphabets)
        dst_char = self.alphabets[0]
        # random sample reference characters
        random.shuffle(self.alphabets)
        ref_chars = self.alphabets[:self.nencode]
        if self.prompts:
            random.shuffle(self.prompts)
            prompt = self.prompts[0].strip() % (dst_char)
        else:
            prompt = "The Chinese character '%s'." %(dst_char)

        refs = []
        for ref_char in ref_chars:
            ref_filename = os.path.join(style, '%s.png' % (ref_char))
            ref_image = Image.open(ref_filename).convert('RGB')
            if self.dim:
                ref_image = ref_image.resize(self.dim)
            refs.append(np.array(ref_image))
        refs = np.concatenate(refs, 2)
        refs = (refs.astype(np.float32) / 127.5) - 1.0

        dst_filename = os.path.join(style, '%s.png' % (dst_char))
        if self.dim:
            dst_image = np.array(Image.open(dst_filename).convert('RGB').resize(self.dim))
        else:
            dst_image = np.array(Image.open(dst_filename).convert('RGB'))
        dst_image = (dst_image.astype(np.float32) / 127.5) - 1.0

        random_num = random.randint(0, 100)
        if random_num < self.prob_mask_ref:
            refs = refs * 0

        return dict(jpg=dst_image, txt=prompt, ref=refs, ref_chars=ref_chars)


class LetterFewshotTestDataset(data.Dataset):
    def __init__(self, root='../../../../datasets/font_datasets/SEPARATE/Capitals_colorGrad64/train',
                 few_num=3, nencode=1, test_nums=None, suffix='_all', dim=(64, 64), is_fixed_prompt=True):
        # suffix = '_unseen-greek'
        self.root = root
        self.suffix = suffix
        self.data = self.get_lines()
        self.few_num = few_num
        self.nencode = nencode
        self.dim = dim
        assert (self.nencode <= self.few_num)

        self.data = self.data[:test_nums]
        print('the number of test dataset', len(self.data))
        if not is_fixed_prompt:
            self.prompts = self.get_prompts()
        else:
            self.prompts = None

    def get_prompts(self):
        prompts_path = 'chinese_prompts.txt'
        if not os.path.exists(prompts_path):
            prompts_path = './data/chinese_prompts.txt'
        if not os.path.exists(prompts_path):
            prompts_path = '../data/chinese_prompts.txt'
        with open(prompts_path, 'r') as f:
            prompts = f.readlines()
        return prompts

    def parse_line(self, content):
        line = []
        prompt, ref_chars_line, dst_chr = content.split('&')
        line.append(prompt.strip())
        ref_chars = []
        for ref_char in ref_chars_line.split('+'):
            if len(ref_char) > 4:
                ref_chars.append(ref_char.strip())
        line.append(ref_chars)
        line.append(dst_chr.strip())
        return line

    def get_lines(self):
        file_name = '%s%s.txt' %(os.path.basename(self.root), self.suffix)
        if not os.path.exists(file_name):
            file_name = os.path.join('./data', file_name)
        if not os.path.exists(file_name):
            file_name = os.path.join('..', file_name)
        if not os.path.exists(file_name):
            print('%s is not exists' %(file_name))
            assert os.path.exists(file_name)
        lines = []
        with open(file_name, 'r') as f:
            line = f.readline()
            while line:
                line = self.parse_line(line)
                lines.append(line)
                line = f.readline()
        return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        k_v_dict = {}
        prompt, ref_chars, dst_char = self.data[idx]
        refs = []
        for ref_char in ref_chars:
            ref_filename = os.path.join(self.root, ref_char)
            ref_image = Image.open(ref_filename).convert('RGB')
            if self.dim:
                ref_image = ref_image.resize(self.dim)
            refs.append(np.array(ref_image))
        refs = np.concatenate(refs, 2)
        refs = (refs.astype(np.float32) / 127.5) - 1.0
        k_v_dict['ref'] = refs

        dst_filename = os.path.join(self.root, dst_char)
        if os.path.exists(dst_filename):
            if self.dim:
                dst_image = np.array(Image.open(dst_filename).convert('RGB').resize(self.dim))
            else:
                dst_image = np.array(Image.open(dst_filename).convert('RGB'))
            dst_image = (dst_image.astype(np.float32) / 127.5) - 1.0
            k_v_dict['jpg'] = dst_image

        if self.prompts is not None:
            random.shuffle(self.prompts)
            prompt = self.prompts[0].strip() % (prompt[-2])
        else:
            prompt = "The Chinese character '%s'." % (prompt[-2])
        k_v_dict['txt'] = prompt
        k_v_dict['ref_chars'] = ref_chars
        return k_v_dict


def check_data(dict_data):
    for k in dict_data.keys():
        v = dict_data[k]
        if isinstance(v, np.ndarray):
            print('check', k, v.shape, v.min(), v.max())
        elif isinstance(v, str):
            print('check', k, v)
        elif isinstance(v, float):
            print('check', k, v)
        elif isinstance(v, list):
            print('check', k, v)
        else:
            assert 1 == 2


if __name__ == '__main__':
    root = '../../../../datasets/font_datasets/SEPARATE/chinese_100/train'
    test_nums = None
    # dataset = LetterFewshotDataset(root, test_nums=test_nums)
    dataset = LetterFewshotTestDataset(os.path.join(root, '../test'))

    item = dataset[0]
    print('datasets len', len(dataset))
    print('all keys:', item.keys())
    print('text:', item['txt'])
    check_data(item)
