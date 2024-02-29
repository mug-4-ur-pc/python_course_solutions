import numpy as np


def _drt_nearest(img, drt, params):
    p = params.p_start
    for k in range(params.p_amount):
        alpha = p
        tau = params.tau_start
        for h in range(params.tau_amount):
            beta = tau

            m_min, m_max = 0, img.shape[1] - 1
            if alpha > 0:
                m_min = max(m_min, np.ceil((-beta - 0.5) / alpha))
                m_max = min(m_max, np.floor((img.shape[0] - 0.5 - beta) / alpha))
            elif alpha < 0:
                m_min = max(m_min, np.ceil((img.shape[0] - 0.5 - beta) / alpha))
                m_max = min(m_max, np.floor((-beta - 0.5) / alpha))

            m = np.arange(m_min, m_max + 1, dtype=int)
            n = np.round(alpha * m + beta).astype(int)
            res = img[n, m]

            drt[h, k] = res.sum()
            tau += params.dtau

        p += params.dp


def _drt_linear(img, drt, params):
    p = params.p_start
    for k in range(params.p_amount):
        alpha = p
        tau = params.tau_start
        for h in range(params.tau_amount):
            beta = tau

            m_min, m_max = 0, img.shape[1] - 1
            if alpha > 0:
                m_min = max(m_min, np.ceil(-beta / alpha))
                m_max = min(m_max, np.floor((img.shape[0] - 2 - beta) / alpha))
            elif alpha < 0:
                m_min = max(m_min, np.ceil((img.shape[0] - 2 - beta) / alpha))
                m_max = min(m_max, np.floor(-beta / alpha))

            m = np.arange(m_min, m_max + 1, dtype=int)
            nfloat = alpha * m + beta
            n = np.floor(nfloat).astype(int)
            w = nfloat - n
            res = img[n, m] * (1 - w) + img[n + 1, m] * w

            drt[h, k] = res.sum()
            tau += params.dtau

        p += params.dp


def _drt_sinc(img, drt, params):
    p = params.p_start
    for k in range(params.p_amount):
        alpha = min(1, 1 / np.abs(p))
        tau = params.tau_start
        for h in range(params.tau_amount):
            gamma = p * (np.arange(img.shape[1]) + tau) * np.pi
            gamma = gamma - np.pi * np.expand_dims(np.arange(img.shape[0]), axis=1)

            res = np.ones_like(img)
            gamma_not_0 = gamma != 0
            res[gamma_not_0] = np.sin(gamma[gamma_not_0] * alpha) / gamma[gamma_not_0]
            res *= img

            drt[h, k] = res.sum()
            tau += params.dtau

        p += params.dp


class DRTParams:
    def __init__(
        self,
        *,
        p_amount=101,
        tau_amount=101,
        p_start=-1.0,
        tau_start=0.0,
        dp=1 / 1024,
        dtau=0.0,
    ):
        self.p_amount = p_amount
        self.tau_amount = tau_amount
        self.p_start = p_start
        self.tau_start = tau_start
        self.dp = dp
        self.dtau = 1 if dtau <= 0 else dtau


def drt(
    img,
    interpolation="linear",
    *,
    params=None,
    p_amount=0,
    tau_amount=0,
    p_start=-1.0,
    tau_start=0.0,
    dp=0.0,
    dtau=0.0,
    drt_buffer=None,
):
    tau_amount = tau_amount if tau_amount > 0 else img.shape[0]
    dp = dp if dp > 0 else 1 / (img.shape[1] - 1)
    p_amount = p_amount if p_amount > 0 else int(np.round(2 / dp)) + 1
    if params is None:
        params = DRTParams(
            p_amount=p_amount,
            tau_amount=tau_amount,
            p_start=p_start,
            tau_start=tau_start,
            dp=dp,
            dtau=dtau,
        )

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

    return drt_buffer
