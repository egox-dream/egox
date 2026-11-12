"""Sensor perturbations of the out-of-distribution trials, applied to the rendered images.

Each OOD trial fixes exposure, white balance, contrast, saturation and hue per camera and
one corruption profile (blur, pixelation, Gaussian noise, JPEG). Noise is reseeded from the
trial seed, camera, sample index and stereo eye, so a rerun sees identical pixels. The
policy and the review video see the same processed images. Ported unchanged from the
simulator of the paper's benchmark.
"""

import hashlib
import json

import cv2
import numpy as np

CORRUPTION_ALGORITHM = 'egox_sensor_corruptions_v1'


def content_seed(value):
    text = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def photometric_lut(exposure_ev, white_balance_scale, contrast):
    """Exposure and white balance in linear light, then display-space contrast, per input byte."""
    values = np.broadcast_to(np.arange(256)[:, None], (256, 3)).astype(np.float32) / 255.
    values = np.where(values <= .04045, values / 12.92, ((values + .055) / 1.055) ** 2.4)
    values *= float(2. ** exposure_ev)
    values *= np.asarray(white_balance_scale, dtype=np.float32)
    values = np.where(values <= .0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - .055)
    return .5 + (values - .5) * contrast


def shift_hue(image, shift):
    if not shift:
        return image.copy()
    hsv = cv2.cvtColor(image.astype(np.float32) / 255., cv2.COLOR_RGB2HSV)
    hsv[..., 0] = np.mod(hsv[..., 0] + float(shift) * 360., 360.)
    return np.rint(np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0., 1.) * 255.).astype(np.uint8)


def corrupt(image, sample, seed, main_eye):
    """Blur, pixelation, noise and JPEG at a reference resolution; image geometry is unchanged."""
    height, width = image.shape[:2]
    reference_width = min(width, sample['reference_width_px']['main_eye' if main_eye else 'wrist'])
    reference_height = max(1, round(height * reference_width / width))
    values = image.astype(np.float32)
    sigma = float(sample['blur_sigma_px']) * width / reference_width
    if sigma > 0.:
        values = cv2.GaussianBlur(values, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    scale = float(sample['pixelation_scale'])
    if scale < 1.:
        small = cv2.resize(values, (max(1, round(reference_width * scale)), max(1, round(reference_height * scale))),
                           interpolation=cv2.INTER_AREA)
        values = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    std = float(sample['noise_std_dn'])
    if std > 0.:
        noise = np.random.Generator(np.random.PCG64(seed)).normal(0., std, (reference_height, reference_width, 3))
        noise = noise.astype(np.float32)
        if (reference_height, reference_width) != (height, width):
            noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)
        shadow_gain = float(sample.get('shadow_noise_gain', 1.))
        if shadow_gain != 1.:
            luminance = (values * np.array([.2126, .7152, .0722], dtype=np.float32)).sum(-1, keepdims=True) / 255.
            noise *= 1. + (shadow_gain - 1.) * (1. - np.clip(luminance, 0., 1.)) ** 2
        values += noise
    result = np.rint(np.clip(values, 0., 255.)).astype(np.uint8)
    quality = int(sample['jpeg_quality'])
    if quality < 100:
        reference = cv2.resize(result, (reference_width, reference_height), interpolation=cv2.INTER_AREA)
        _, encoded = cv2.imencode('.jpg', cv2.cvtColor(reference, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        decoded = cv2.cvtColor(cv2.imdecode(encoded, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        result = cv2.resize(decoded, (width, height), interpolation=cv2.INTER_LINEAR)
    return result


def postprocess(image, sensor, row, camera):
    """Perturb one RGB uint8 camera output (`main`, `left_wrist`, `right_wrist`) at 30 Hz sample `row`.

    `sensor` is the trial's sensor entry: `seed`, `views` (optical eyes per output) and the
    per-camera `image` samples.
    """
    sample = sensor['image'].get('camera_samples', {}).get(camera, sensor['image'])
    lut = photometric_lut(sample['exposure_ev'], tuple(sample['white_balance_scale']), sample['contrast'])
    values = cv2.LUT(image, lut.reshape(256, 1, 3))
    luminance = values[..., 0] * np.float32(.2126)
    luminance += values[..., 1] * np.float32(.7152)
    luminance += values[..., 2] * np.float32(.0722)
    luminance = luminance[..., None]
    values -= luminance
    values *= sample['saturation']
    values += luminance
    values *= 255.
    if sample['noise_std_dn']:
        rng = np.random.Generator(np.random.PCG64(content_seed({'seed': sensor['seed'], 'camera': camera, 'row': row})))
        values += rng.normal(0., sample['noise_std_dn'], size=values.shape).astype(np.float32)
    np.clip(values, 0., 255., out=values)
    np.rint(values, out=values)
    result = shift_hue(values.astype(np.uint8), sample['hue_shift'])
    eyes = sensor['views'][camera]  # stereo eyes are corrupted separately
    parts = [corrupt(part, sample['corruption'], content_seed(
        {'seed': sensor['seed'], 'camera': camera, 'row': row, 'eye': eye, 'algorithm': CORRUPTION_ALGORITHM}),
        main_eye=camera == 'main') for eye, part in enumerate(np.split(result, eyes, axis=1))]
    return parts[0] if eyes == 1 else np.concatenate(parts, axis=1)
