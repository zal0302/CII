import numpy as np
import numbers, collections
import random
import torch
from PIL import Image
from torchvision.transforms import functional as F

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX',
}

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

class ComposePair(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, gt, edge=None):
        if edge is None:
            for t in self.transforms:
                img, gt = t(img, gt)
            return img, gt
        else:
            for t in self.transforms:
                img, gt, edge = t(img, gt, edge)
            return img, gt, edge

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class ResizePair(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.abc.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, gt, edge=None):
        if edge is None:
            return F.resize(img, self.size, self.interpolation), F.resize(gt, self.size, self.interpolation)
        else:
            return F.resize(img, self.size, self.interpolation), F.resize(gt, self.size, self.interpolation), F.resize(edge, self.size, self.interpolation)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class RandomBorderCropPair(object):
    """Crop out the border pixels of the given image.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    Args:
        size (sequence or int): Desired number of pixels from the border to be cropped out. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a tuple or list of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If tuple of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a tuple of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a tuple or
            list of length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (int or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
            Mode symmetric is not yet supported for Tensor inputs.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    @staticmethod
    def get_params(img, border_size):
        """Get parameters for ``crop`` for a random border crop.
        Args:
            img (PIL Image or Tensor): Image to be cropped.
            border_size (tuple): Expected border size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random border crop.
        """
        w, h = _get_image_size(img)
        th, tw = border_size

        i_1 = torch.randint(0, th, size=(1, )).item()
        i_2 = torch.randint(0, th, size=(1, )).item()
        j_1 = torch.randint(0, tw, size=(1, )).item()
        j_2 = torch.randint(0, tw, size=(1, )).item()

        th = h - i_1 - i_2
        tw = w - j_1 - j_2

        return i_1, j_1, th, tw

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        super().__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif isinstance(size, Sequence) and len(size) == 1:
            self.size = (size[0], size[0])
        else:
            if len(size) != 2:
                raise ValueError("Please provide only two dimensions (h, w) for size.")

            # cast to tuple for torchscript
            self.size = tuple(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img, gt, edge=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped.
        Returns:
            PIL Image or Tensor: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            gt = F.pad(gt, self.padding, self.fill, self.padding_mode)
            if edge is not None:
                edge = F.pad(edge, self.padding, self.fill, self.padding_mode)

        width, height = _get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            gt = F.pad(gt, padding, self.fill, self.padding_mode)
            if edge is not None:
                edge = F.pad(edge, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            gt = F.pad(gt, padding, self.fill, self.padding_mode)
            if edge is not None:
                edge = F.pad(edge, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        if edge is None:
            return F.crop(img, i, j, h, w), F.crop(gt, i, j, h, w)
        else:
            return F.crop(img, i, j, h, w), F.crop(gt, i, j, h, w), F.crop(edge, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


class RandomHorizontalFlipPair(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, gt, edge=None):
        if random.random() < self.p:
            img = F.hflip(img)
            gt = F.hflip(gt)
            if edge is not None:
                edge = F.hflip(edge)
        if edge is None:
            return img, gt
        else:
            return img, gt, edge

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotationPair(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (n-tuple or int or float): Pixel fill value for area outside the rotated
            image. If int or float, the value is used for all bands respectively.
            Defaults to 0 for all bands. This option is only available for ``pillow>=5.2.0``.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """
    def __init__(self, degrees, resample=False, expand=False, center=None, fill=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, gt, edge=None):
        angle = self.get_params(self.degrees)

        if edge is None:
            return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill), F.rotate(gt, angle, self.resample, self.expand, self.center, self.fill)
        else:
            return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill), F.rotate(gt, angle, self.resample, self.expand, self.center, self.fill), F.rotate(edge, angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string        

class RandomCropPair(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
    """

    def __init__(self, percent=0.125):
        self.percent = percent

    @staticmethod
    def get_params(img, percent):
        w, h = _get_image_size(img)
        randw   = np.random.randint(w*percent)
        randh   = np.random.randint(h*percent)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        i, j, th, tw = offseth, offsetw, h+offseth-randh, w+offsetw-randw
        return i, j, th, tw

    def __call__(self, img, gt, edge=None):
        i, j, h, w = self.get_params(img, self.percent)

        if edge is None:
            return F.crop(img, i, j, h, w), F.crop(gt, i, j, h, w)
        else:
            return F.crop(img, i, j, h, w), F.crop(gt, i, j, h, w), F.crop(edge, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(percent={0})'.format(self.percent)

class ToTensorPair(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic1, pic2, pic3=None):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if pic3 is None:
            return F.to_tensor(pic1), F.to_tensor(pic2)
        else:
            return F.to_tensor(pic1), F.to_tensor(pic2), F.to_tensor(pic3)

    def __repr__(self):
        return self.__class__.__name__ + '()'

