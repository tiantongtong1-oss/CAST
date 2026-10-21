"""Source-only resolution degradation before the existing weak transform."""
import math
import random
import numpy as np
from PIL import Image

class RandomSourceDownsample:
    def __init__(self, probability=0.5, size=48):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('probability must be finite in [0, 1]')
        if isinstance(size, bool) or int(size) != size or size < 1:
            raise ValueError('size must be a positive integer')
        self.probability, self.size = probability, int(size)

    def __call__(self, image):
        if self.probability == 0:
            return image  # Disabled baseline consumes no RNG.
        if random.random() >= self.probability:
            return image
        image = Image.fromarray(np.asarray(image))
        width, height = image.size
        if min(width, height) <= self.size:
            return np.asarray(image).copy()
        scale = self.size / min(width, height)
        small = image.resize((round(width*scale), round(height*scale)), Image.Resampling.BILINEAR)
        return np.asarray(small.resize((width,height), Image.Resampling.BILINEAR)).copy()
