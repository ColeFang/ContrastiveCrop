from torchvision.transforms import Compose
from PIL import ImageFilter
import random
import numpy as np


class CCompose(Compose):
    def __call__(self, x):  # x: [sample, box]
        img = self.transforms[0](*x)
        for t in self.transforms[1:]:
            img = t(img)
        return img


class MultiViewTransform:
    """Create multiple views of the same image"""
    def __init__(self, transform, num_views=2):
        if not isinstance(transform, (list, tuple)):
            transform = [transform for _ in range(num_views)]
        self.transforms = transform

    def __call__(self, x):
        views = [t(x) for t in self.transforms]
        return views

class MixupTransform:
    """Create multiple views of the same image"""
    def __init__(self, transform, num_views=2):
        if not isinstance(transform, (list, tuple)):
            transform = [transform for _ in range(num_views)]
        self.transforms = transform

    def __call__(self, alpha, x):
        #mixup_lambda = np.random.beta(alpha, alpha)
        mixup_lambda = alpha
        #print("using beta distribution: ", mixup_lambda)
        if mixup_lambda < 0:
            mixup_lambda = 0
        elif mixup_lambda > 1:
            mixup_lambda = 1
        views = [t(x) for t in self.transforms]
        views[0] = mixup_lambda * views[1] + (1 - mixup_lambda) * views[0]
        return views


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

