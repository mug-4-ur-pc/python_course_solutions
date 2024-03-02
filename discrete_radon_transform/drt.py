from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt


def _drt_nearest(img, drt, params):
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


def _drt_linear(img, drt, params):
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


def _drt_sinc(img, drt, params):
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


def _prepare_for_drt(img):
    if not isinstance(img, np.ndarray):
        raise TypeError("Image must be a numpy array")

    match img.ndim:
        case 2:
            return img
        case 3:
            match img.shape[2]:
                case 3:
                    return img.sum(axis=2)
                case 4:
                    return img[..., :3].sum(axis=2)
                case _:
                    raise TypeError(f"Image with unknown shape {img.shape}")
        case _:
            raise TypeError(f"Image with unknown shape {img.shape}")


@dataclass
class DRTParams:
    p_amount: int
    tau_amount: int
    p_min: float
    p_max: float
    tau_min: float
    tau_max: float
    dp: float
    dtau: float


def drt(
    img,
    interpolation="linear",
    *,
    params: Optional[DRTParams] = None,
    p_min=-1.0,
    p_max=1.0,
    tau_min=0.0,
    tau_max=-1.0,
    dp=0.0,
    dtau=0.0,
    drt_buffer: Optional[npt.NDArray[np.float64]] = None,
):
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
    img,
    interpolation="linear",
    *,
    dp=0.0,
    dtau=0.0,
    drt_buffers: Optional[list[Optional[npt.NDArray[np.float64]]]] = None,
):
    img = _prepare_for_drt(img)
    if drt_buffers == None:
        drt_buffers = [None, None]

    params: Optional[list[DRTParams]] = [None, None]
    drt_buffers[0], params[0] = drt(
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

    drt_buffers[1], params[1] = drt(
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

    return drt_buffers, params
