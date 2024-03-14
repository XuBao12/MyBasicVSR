import numpy as np
from .misc import tensor2img
from .utils import bgr2ycbcr
import cv2


def evaluate(output, gt, metrics):
    """Evaluation function.

    If the output contains multiple frames, we compute the metric
    one by one and take an average.

    Args:
        output (Tensor): Model output with shape (n, t, c, h, w).
        gt (Tensor): GT Tensor with shape (n, t, c, h, w).
        metrics (dict): dict in dict. {'metric': {'kwargs': ...}}.
            Kwargs is keyword arguments in cal_psnr() and cal_ssim() like crop_border, input_order and convert_to

    Returns:
        dict: Evaluation results. {'psnr': value, 'ssim': value}.
    """
    allowed_metrics = {"psnr": cal_psnr, "ssim": cal_ssim}
    allowed_kwargs = ["crop_border", "input_order", "convert_to"]
    args = metrics.copy()

    for metric, kwargs in args.items():
        if metric.lower() not in allowed_metrics.keys():
            raise KeyError(f"Metric {metric} is not allowed")
        if kwargs:
            for k, v in kwargs.items():
                if k not in allowed_kwargs:
                    raise KeyError(f"Parameter {k} is not allowed")

    eval_result = dict()
    for metric, kwargs in args.items():
        if output.ndim == 5:  # a sequence: (1, t, c, h, w)
            avg = []
            for i in range(0, output.size(1)):
                output_i = tensor2img(output[:, i, :, :, :])
                gt_i = tensor2img(gt[:, i, :, :, :])
                avg.append(allowed_metrics[metric](output_i, gt_i, **kwargs))
            eval_result[metric] = np.mean(avg)
        elif output.ndim == 4:  # an image: (1, c, h, w), for Vimeo-90K-T
            output_img = tensor2img(output)
            gt_img = tensor2img(gt)
            value = allowed_metrics[metric](output_img, gt_img, **kwargs)
            eval_result[metric] = value

    return eval_result


def evaluate_no_avg(output, gt, metrics):
    """Evaluation function.

    If the output contains multiple frames, we compute the metric
    one by one and don't take an average.

    Args:
        output (Tensor): Model output with shape (n, t, c, h, w).
        gt (Tensor): GT Tensor with shape (n, t, c, h, w).
        metrics (dict): dict in dict. {'metric': {'kwargs': ...}}.
            Kwargs is keyword arguments in cal_psnr() and cal_ssim() like crop_border, input_order and convert_to

    Returns:
        dict: Evaluation results. {'psnr': value, 'ssim': value}.
    """
    allowed_metrics = {"psnr": cal_psnr, "ssim": cal_ssim}
    allowed_kwargs = ["crop_border", "input_order", "convert_to"]
    args = metrics.copy()

    for metric, kwargs in args.items():
        if metric.lower() not in allowed_metrics.keys():
            raise KeyError(f"Metric {metric} is not allowed")
        if kwargs:
            for k, v in kwargs.items():
                if k not in allowed_kwargs:
                    raise KeyError(f"Parameter {k} is not allowed")

    eval_result = dict()
    for metric, kwargs in args.items():
        if output.ndim == 5:  # a sequence: (1, t, c, h, w)
            avg = []
            for i in range(0, output.size(1)):
                output_i = tensor2img(output[:, i, :, :, :])
                gt_i = tensor2img(gt[:, i, :, :, :])
                avg.append(allowed_metrics[metric](output_i, gt_i, **kwargs))
            eval_result[metric] = avg
        elif output.ndim == 4:  # an image: (1, c, h, w), for Vimeo-90K-T
            output_img = tensor2img(output)
            gt_img = tensor2img(gt)
            value = allowed_metrics[metric](output_img, gt_img, **kwargs)
            eval_result[metric] = value

    return eval_result


def cal_psnr(img1, img2, crop_border=0, input_order="HWC", convert_to=None, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: psnr result.
    """
    crop_border = kwargs["crop_border"] if "crop_border" in kwargs else crop_border
    input_order = kwargs["input_order"] if "input_order" in kwargs else input_order
    convert_to = kwargs["convert_to"] if "convert_to" in kwargs else convert_to

    assert (
        img1.shape == img2.shape
    ), f"Image shapes are different: {img1.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are "
            '"HWC" and "CHW"'
        )
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if isinstance(convert_to, str) and convert_to.lower() == "y":
        img1 = bgr2ycbcr(img1 / 255.0, y_only=True) * 255.0
        img2 = bgr2ycbcr(img2 / 255.0, y_only=True) * 255.0
    elif convert_to is not None:
        raise ValueError("Wrong color model. Supported values are " '"Y" and None.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2) ** 2)
    if mse_value == 0:
        return float("inf")

    psnr = 20.0 * np.log10(255.0 / np.sqrt(mse_value))
    return psnr


def cal_ssim(img1, img2, crop_border=0, input_order="HWC", convert_to=None, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: ssim result.
    """
    crop_border = kwargs["crop_border"] if "crop_border" in kwargs else crop_border
    input_order = kwargs["input_order"] if "input_order" in kwargs else input_order
    convert_to = kwargs["convert_to"] if "convert_to" in kwargs else convert_to

    assert (
        img1.shape == img2.shape
    ), f"Image shapes are different: {img1.shape}, {img2.shape}."
    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are "
            '"HWC" and "CHW"'
        )
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if isinstance(convert_to, str) and convert_to.lower() == "y":
        img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
        img1 = bgr2ycbcr(img1 / 255.0, y_only=True) * 255.0
        img2 = bgr2ycbcr(img2 / 255.0, y_only=True) * 255.0
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
    elif convert_to is not None:
        raise ValueError("Wrong color model. Supported values are " '"Y" and None')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def reorder_image(img, input_order="HWC"):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ["HWC", "CHW"]:
        raise ValueError(
            f"Wrong input_order {input_order}. Supported input_orders are "
            '"HWC" and "CHW"'
        )
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == "CHW":
        img = img.transpose(1, 2, 0)
    return img
