import os
import torch
from main.dataloader.utils.transforms import test_transform
from PIL import Image
import os
from typing import Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame

from main.utils.jsonData import JsonData
from main.models import get_model
from main.utils import *


class LatexOCR():
    '''Get a prediction of an image in the easiest way'''

    image_resizer = None
    last_pic = None
    model_config: JsonData = None

    def __init__(self, model_config, vocab_file,  encoder_structure='hybrid', resume_path=None, no_resize=True, resize_model_path=None):
        self.model_config = model_config
        self.vocab_file = vocab_file
        self.encoder_structure = encoder_structure
        self.no_resize = no_resize
        self.resize_model_path = resize_model_path
        self.model_config.no_resize = no_resize
        self.model_config.max_dimensions = [
            self.model_config.max_width, self.model_config.max_height]
        self.model_config.min_dimensions = [
            self.model_config.min_width, self.model_config.min_height]
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_config.device = self.device
        self.model = get_model(self.encoder_structure,
                               self.device, self.model_config)
        if resume_path is not None:
            print('Accessing Resume PATH: {} ...\n'.format(resume_path))
            self.model.load_state_dict(torch.load(
                resume_path, map_location=self.device))
        self.model.eval()

        if self.resize_model_path is not None and not self.no_resize:
            self.image_resizer = ResNetV2(layers=[2, 3, 3], num_classes=max(self.model_config.max_dimensions)//32, global_pool='avg', in_chans=1, drop_rate=.05,
                                          preact=True, stem_type='same', conv_layer=StdConv2dSame).to(self.device)
            self.image_resizer.load_state_dict(torch.load(
                self.resize_model_path, map_location=self.device))
            self.image_resizer.eval()
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=self.vocab_file)

    def __call__(self, img=None, resize=True) -> str:
        """Get a prediction from an image

        Args:
            img (Image, optional): Image to predict. Defaults to None.
            resize (bool, optional): Whether to call the resize model. Defaults to True.

        Returns:
            str: predicted Latex code
        """
        if type(img) is bool:
            img = None
        if img is None:
            if self.last_pic is None:
                return ''
            else:
                print('\nLast image is: ', end='')
                img = self.last_pic.copy()
        else:
            self.last_pic = img.copy()
        img = self.minmax_size(
            pad(img), self.model_config.max_dimensions, self.model_config.min_dimensions)
        if (self.image_resizer is not None and not self.no_resize) and resize:
            with torch.no_grad():
                input_image = img.convert('RGB').copy()
                r, w, h = 1, input_image.size[0], input_image.size[1]
                for _ in range(10):
                    h = int(h * r)  # height to resize
                    img = pad(self.minmax_size(input_image.resize((w, h), Image.Resampling.BILINEAR if r >
                              1 else Image.Resampling.LANCZOS), self.model_config.max_dimensions, self.model_config.min_dimensions))
                    t = test_transform(image=np.array(img.convert('RGB')))[
                        'image'][:1].unsqueeze(0)
                    w = (self.image_resizer(
                        t.to(self.device)).argmax(-1).item()+1)*32
                    if (w == img.size[0]):
                        break
                    r = w/img.size[0]
        else:
            img = np.array(pad(img).convert('RGB'))
            t = test_transform(image=img)['image'][:1].unsqueeze(0)
        im = t.to(self.device)

        dec = self.model.generate(
            im.to(self.device), temperature=self.model_config.get('temperature', .25))
        pred = post_process(token2str(dec, self.tokenizer)[0])
        return pred

    def minmax_size(self, img: Image, max_dimensions: Tuple[int, int] = None, min_dimensions: Tuple[int, int] = None) -> Image:
        """Resize or pad an image to fit into given dimensions

        Args:
            img (Image): Image to scale up/down.
            max_dimensions (Tuple[int, int], optional): Maximum dimensions. Defaults to None.
            min_dimensions (Tuple[int, int], optional): Minimum dimensions. Defaults to None.

        Returns:
            Image: Image with correct dimensionality
        """
        if max_dimensions is not None:
            ratios = [a/b for a, b in zip(img.size, max_dimensions)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size)//max(ratios)
                img = img.resize(size.astype(int), Image.BILINEAR)
        if min_dimensions is not None:
            # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
            padded_size = [max(img_dim, min_dim)
                           for img_dim, min_dim in zip(img.size, min_dimensions)]
            if padded_size != list(img.size):  # assert hypothesis
                padded_im = Image.new('L', padded_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img
