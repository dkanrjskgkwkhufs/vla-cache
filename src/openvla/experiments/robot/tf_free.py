"""TensorFlow-free replacements for the two image preprocessing functions used in LIBERO eval.

TensorFlow segfaults on Thor next to triton / MuJoCo EGL, so the TF ops are re-implemented with PIL/NumPy:
  resize_image  : tf.image.encode_jpeg (q95) -> tf.io.decode_image -> tf.image.resize(lanczos3, antialias=True)
  process_image : tf.image.crop_and_resize (bilinear) of the centre crop with area `crop_scale`, back to 224x224
"""

import functools
import io

import numpy as np
import simplejpeg
from PIL import Image


def _lanczos3(x):
    x = np.abs(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.sinc(x) * np.sinc(x / 3.0)
    return np.where(x < 3.0, y, 0.0)


@functools.lru_cache(maxsize=None)
def _resize_weights(in_size, out_size):
    """Dense (out_size, in_size) weight matrix of tf.image.resize(lanczos3, antialias=True) along one axis."""
    scale = out_size / in_size
    inv_scale = 1.0 / scale
    kernel_scale = max(inv_scale, 1.0)  # antialias: widen the kernel when downscaling
    radius = 3.0 * kernel_scale
    weights = np.zeros((out_size, in_size), dtype=np.float64)
    for i in range(out_size):
        sample = (i + 0.5) * inv_scale
        start = max(int(np.ceil(sample - radius - 0.5)), 0)
        end = min(int(np.floor(sample + radius - 0.5)), in_size - 1)
        src = np.arange(start, end + 1)
        w = _lanczos3((src + 0.5 - sample) / kernel_scale)
        total = w.sum()
        if abs(total) > 1000.0 * np.finfo(np.float32).tiny:
            w = w / total
        weights[i, start : end + 1] = w
    return weights.astype(np.float32)


def resize_image(img, resize_size):
    """Drop-in for experiments.robot.libero.libero_utils.resize_image (uint8 HWC in, uint8 HWC out)."""
    assert isinstance(resize_size, tuple)
    # Encoder/decoder pair checked against TF 2.21 on 50 LIBERO frames: PIL encode (slow DCT, 4:2:0) followed by
    # a fast-DCT decode reproduces tf.image.encode_jpeg -> tf.io.decode_image exactly; other pairs match ~35-55%.
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="JPEG", quality=95, subsampling=2)
    img = simplejpeg.decode_jpeg(buf.getvalue(), colorspace="RGB", fastdct=True, fastupsample=False)
    out_h, out_w = resize_size
    wh = _resize_weights(img.shape[0], out_h)
    ww = _resize_weights(img.shape[1], out_w)
    x = img.astype(np.float32)
    x = np.einsum("oh,hwc->owc", wh, x)
    x = np.einsum("pw,owc->opc", ww, x)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def crop_and_resize(image, crop_scale, out_size=(224, 224)):
    """tf.image.crop_and_resize (bilinear) for one float32 HWC image and a centred box of area `crop_scale`."""
    h, w = image.shape[:2]
    side = np.float32(np.clip(np.sqrt(np.float32(crop_scale)), 0, 1))
    off = (np.float32(1) - side) / np.float32(2)
    y1, x1, y2, x2 = off, off, off + side, off + side
    out_h, out_w = out_size

    def axis(lo, hi, n_in, n_out):
        step = (hi - lo) * np.float32(n_in - 1) / np.float32(n_out - 1)
        pos = lo * np.float32(n_in - 1) + np.arange(n_out, dtype=np.float32) * step
        low = np.floor(pos).astype(np.int64)
        high = np.ceil(pos).astype(np.int64)
        return low, high, (pos - low).astype(np.float32)

    top, bottom, ly = axis(y1, y2, h, out_h)
    left, right, lx = axis(x1, x2, w, out_w)
    lx = lx[None, :, None]
    ly = ly[:, None, None]
    tl = image[top][:, left]
    tr = image[top][:, right]
    bl = image[bottom][:, left]
    br = image[bottom][:, right]
    upper = tl + (tr - tl) * lx
    lower = bl + (br - bl) * lx
    return upper + (lower - upper) * ly


def process_image(image, crop_scale=0.9, batch_size=1):
    """Drop-in for experiments.robot.openvla_utils.process_image (PIL in, PIL RGB out)."""
    x = np.asarray(image).astype(np.float32) / np.float32(255.0)  # tf.image.convert_image_dtype(uint8 -> float32)
    x = crop_and_resize(x, crop_scale)
    x = np.clip(x, 0, 1)
    x = np.minimum(x * np.float32(255.5), np.float32(255.0)).astype(np.uint8)  # convert_image_dtype(saturate=True)
    return Image.fromarray(x).convert("RGB")