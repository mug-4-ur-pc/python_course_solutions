"""Discrete radon transform implementation."""

from dataclasses import dataclass
from typing import Optional

import cv2 as cv
import numpy as np
from numpy.typing import NDArray


@dataclass
class DRTParams:
    """Discrete Radon Transform parameters.

    It specifies properties of the DRT space. The line is defined by
    its slope (p) (defined in radians) and offset (tau) (defined in pixels).

    Attributes:
        p_amount: Amount of possible line slopes.
        tau_amount: Amount of possible offsets.
        p_min: Minimal slope.
        p_max: Maximum slope.
        tau_min: Minimal offset.
        tau_max: Maximum offset.
        dp: Step of slopes.
        dtau: Step of offsets.
    """

    p_amount: int
    tau_amount: int
    p_min: float
    p_max: float
    tau_min: float
    tau_max: float
    dp: float
    dtau: float


def _drt_nearest(
    img: NDArray[np.uint8], drt: NDArray[np.float_], params: DRTParams
) -> None:
    """Apply DRT using Nearest Neighbour Interpolation.

    Note:
        It is the fastest method and less precise method.

    Args:
        img: Input image.
        drt: Buffer where transform result will be written.
        params: Transform space properties.
    """
    p = params.p_min
    for k in range(params.p_amount):
        alpha = p
        tau = params.tau_min
        for h in range(params.tau_amount):
            beta = tau

            m_min, m_max = 0, img.shape[1] - 1
            if alpha > 0:
                m_min = max(m_min, np.ceil((-beta - 0.5) / alpha + 1e-10).astype(int))
                m_max = min(
                    m_max,
                    np.floor((img.shape[0] - 0.5 - beta) / alpha - 1e-10).astype(int),
                )
            elif alpha < 0:
                m_min = max(
                    m_min,
                    np.ceil((img.shape[0] - 0.5 - beta) / alpha + 1e-10).astype(int),
                )
                m_max = min(m_max, np.floor((-beta - 0.5) / alpha - 1e-10).astype(int))

            m = np.arange(m_min, m_max + 1, dtype=int)
            n = np.round(alpha * m + beta).astype(int)
            res = img[n, m]

            drt[h, k] = res.sum()
            tau += params.dtau

        p += params.dp


def _drt_linear(
    img: NDArray[np.uint8], drt: NDArray[np.float_], params: DRTParams
) -> None:
    """Apply DRT using Linear Interpolation.

    Note:
        It shows better accuracy than Nearest Neighbour Interpolation method
        and their performance are comparable.

    Args:
        img: Input image.
        drt: Buffer where transform result will be written.
        params: Transform space properties.
    """
    p = params.p_min
    for k in range(params.p_amount):
        alpha = p
        tau = params.tau_min
        for h in range(params.tau_amount):
            beta = tau

            m_min, m_max = 0, img.shape[1] - 1
            if alpha > 0:
                m_min = max(m_min, np.ceil(-beta / alpha + 1e-10).astype(int))
                m_max = min(
                    m_max,
                    np.floor((img.shape[0] - 1 - beta) / alpha - 1e-10).astype(int),
                )
            elif alpha < 0:
                m_min = max(
                    m_min,
                    np.ceil((img.shape[0] - 1 - beta) / alpha + 1e-10).astype(int),
                )
                m_max = min(m_max, np.floor(-beta / alpha - 1e-10).astype(int))

            m = np.arange(m_min, m_max + 1, dtype=int)
            nfloat = alpha * m + beta
            n = np.floor(nfloat - 1e-10).astype(int)
            w = nfloat - n
            res = img[n, m] * (1 - w) + img[n + 1, m] * w

            drt[h, k] = res.sum()
            tau += params.dtau

        p += params.dp


def _drt_sinc(
    img: NDArray[np.uint8], drt: NDArray[np.float_], params: DRTParams
) -> None:
    """Apply Whittaker-Shannon theorem to recover original function from its discretization.

    Note:
        This method is the most accurate.
        However, it should be avoided because of its complexity.

    Args:
        img: Input image.
        drt: Buffer where transform result will be written.
        params: Transform space properties.
    """
    p = params.p_min
    for k in range(params.p_amount):
        with np.errstate(divide="ignore"):
            alpha = min(1, 1 / np.abs(p))
        tau = params.tau_min
        for h in range(params.tau_amount):
            gamma = p * (np.arange(img.shape[1], dtype=np.float64) + tau) * np.pi
            gamma = gamma - np.pi * np.expand_dims(
                np.arange(img.shape[0], dtype=np.float64), axis=1
            )

            res = np.full_like(img, alpha)
            gamma_not_0 = gamma != 0
            for_sin = (gamma[gamma_not_0] * alpha) % (2 * np.pi)
            q34 = (np.pi / 2 < for_sin) & (for_sin <= 3 * np.pi / 2)
            q2 = (3 * np.pi / 2 < for_sin) & (for_sin <= 2 * np.pi)
            for_sin[q34] = np.pi - for_sin[q34]
            for_sin[q2] -= 2 * np.pi
            res[gamma_not_0] = np.sin(for_sin) / gamma[gamma_not_0]
            res *= img

            drt[h, k] = res.sum()
            tau += params.dtau

        p += params.dp


def _prepare_for_drt(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Check image type and convert it to grayscale.

    Args:
        img: Colorful or grayscale image.

    Returns:
        Grayscale image.

    Raises:
        TypeError: Image has incompatible type or shape.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError("Image must be a numpy array")

    match img.ndim:
        case 2:
            return img
        case 3:
            match img.shape[2]:
                case 3:
                    return img.mean(axis=2).astype(np.uint8)
                case 4:
                    return img[..., :3].mean(axis=2).astype(np.uint8)
                case _:
                    raise TypeError(f"Image with unknown shape {img.shape}")
        case _:
            raise TypeError(f"Image with unknown shape {img.shape}")


def drt(
    img: NDArray[np.uint8],
    interpolation: str = "linear",
    *,
    params: Optional[DRTParams] = None,
    p_min: float = -1.0,
    p_max: float = 1.0,
    tau_min: float = 0.0,
    tau_max: float = -1.0,
    dp: float = 0.0,
    dtau: float = 0.0,
    drt_buffer: Optional[NDArray[np.float_]] = None,
) -> tuple[NDArray[np.float_], DRTParams]:
    """Apply Discrete Radon Transform to image.

    Discrete Radon Transform is an algorithm which is useful for line or points detection.
    The result of this transformation is a space, where all lines are represented
    as a cluster which coordinates x and y are the source line slope in radians and offset in pixels
    respectively.

    Note: It's impossible to detect all lines in the picture because it requires infinity Radon space.
        For this purpose, use `radon_full` method instead.

    Args:
        img: Input image.
        interpolation: Interpolation method:
            `nearest` - nearest interpolation.
            `linear` - linear interpolation;
            `sinc` - interpolate according to Whittaker-Shannon theorem.
        params: DRT parameters. If it is set, all other parameters will be ignored.
        p_min: Minimal slope value. If params is set, this argument will be ignored.
        p_max: Maximum offset value. If params is set, this argument will be ignored.
        tau_min: Minimal offset value. If params is set, this argument will be ignored.
        tau_max: Maximum slope value. If params is set, this argument will be ignored.
        dp: Slope step. If params is set, this argument will be ignored.
        dtau: Offset step. If params is set, this argument will be ignored.
        drt_buffer: Buffer in which the result will be written. If `None`, new one will be
            created.

    Returns:
        A tuple containing a DRT output and its parameters.

    Raises:
        ValueError: Unknown interpolation method.
    """
    dp = dp if dp > 0 else 1 / img.shape[1]
    dtau = dtau if dtau > 0 else 1
    tau_max = tau_max if tau_max >= tau_min else img.shape[0] - 1
    p_amount = int((p_max - p_min) // dp + 1)
    tau_amount = int((tau_max - tau_min) // dtau + 1)

    if params is None:
        params = DRTParams(
            p_amount=p_amount,
            tau_amount=tau_amount,
            p_min=p_min,
            p_max=p_max,
            tau_min=tau_min,
            tau_max=tau_max,
            dp=dp,
            dtau=dtau,
        )

    img = _prepare_for_drt(img)
    if drt_buffer is None:
        drt_buffer = np.zeros([params.tau_amount, params.p_amount], dtype=np.float64)

    if interpolation == "nearest":
        _drt_nearest(img, drt_buffer, params)
    elif interpolation == "linear":
        _drt_linear(img, drt_buffer, params)
    elif interpolation == "sinc":
        _drt_sinc(img, drt_buffer, params)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation}")

    return drt_buffer, params


def full_drt(
    img: NDArray[np.uint8],
    interpolation: str,
    *,
    dp: float = 0.0,
    dtau: float = 0.0,
    drt_buffers: Optional[
        tuple[Optional[NDArray[np.float_]], Optional[NDArray[np.float_]]]
    ] = None,
) -> tuple[tuple[NDArray[np.float_], NDArray[np.float_]], tuple[DRTParams, DRTParams]]:
    """Apply Discrete Radon Transform to image and its transpose.

    Discrete Radon Transform is an algorithm which is useful for line or points detection.
    The result of this transformation is a space, where all lines are represented
    as a cluster which coordinates x and y are the source line slope in radians and offset in pixels
    respectively.

    Note: It's impossible to detect all lines in the picture because it requires infinity Radon space.
        For this purpose, this method.

    Args:
        img: Input image.
        interpolation: Interpolation method:
            `nearest` - nearest interpolation.
            `linear` - linear interpolation;
            `sinc` - interpolate according to Whittaker-Shannon theorem.
        dp: Slope step. If params is set, this argument will be ignored.
        dtau: Offset step. If params is set, this argument will be ignored.
        drt_buffers: Buffers in which the result will be written.
            If `None`, new one will be created.

    Returns:
        A tuple containing a two DRT outputs and theirs parameters.
    """
    img = _prepare_for_drt(img)
    if drt_buffers is None:
        drt_buffers = None, None

    drt1, params1 = drt(
        img,
        interpolation=interpolation,
        p_min=-1.0,
        p_max=1.0,
        tau_min=0.0,
        tau_max=img.shape[0] - 1,
        dp=dp,
        dtau=dtau,
        drt_buffer=drt_buffers[0],
    )

    drt2, params2 = drt(
        img.T,
        interpolation=interpolation,
        p_min=-1.0,
        p_max=1.0,
        tau_min=0.0,
        tau_max=img.shape[1] - 1,
        dp=dp,
        dtau=dtau,
        drt_buffer=drt_buffers[1],
    )

    return (drt1, drt2), (params1, params2)


def find_lines(
    img: NDArray[np.uint8], interpolation: str = "linear", threshold_ratio: float = 0.9
) -> NDArray[np.uint8]:
    """Find lines in image.

    Args:
        img: Input image.
        interpolation: Interpolation method:
            `nearest` - nearest interpolation.
            `linear` - linear interpolation;
            `sinc` - interpolate according to Whittaker-Shannon theorem.
        threshold_ratio: Threshold value to extract peaks from DRT result image.

    Returns:
        An image with founded lines on it.
    """
    drts, _ = full_drt(img, interpolation)

    transpose = True
    out = np.zeros_like(img)
    for drt in drts:
        threshold = drt.max() * threshold_ratio
        drt_thresholded = np.zeros(drt.shape, dtype=np.uint8)
        drt_thresholded[drt >= threshold] = 255

        contours, _ = cv.findContours(
            drt_thresholded, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE
        )
        for c in contours:
            p, tau = np.mean(c, axis=0).squeeze()

            _draw_line(out, p, tau, transpose)

        transpose = False
    return out


def _draw_line(img: NDArray[np.uint8], p: float, tau: float, transpose: bool) -> None:
    """Draw line gained from DRT space.

    Args:
        img: Image where line will be drawn.
        p: Line slope.
        tau: Line offset.
        transpose: Is DRT space transposed.
    """
    x0, y0 = 0, int(tau)
    x1 = img.shape[0] if transpose else img.shape[1]
    y1 = int(p * x1 + tau)
    if transpose:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    cv.line(img, (x0, y0), (x1, y1), (255, 255, 255))
