import numpy as np


def _drt_nearest(img, drt, params):
    pass


def _drt_linear(img, drt, params):
    p = params.p_start
    for curr_k in range(params.k):
        alpha = p * params.dx / params.dy
        tau = params.tau_start
        for curr_h in range(params.h):
            beta = (p * params.x_min + tau - params.y_min) / params.dy

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
            # print(m_min, m_max, alpha, beta, n[0])
            w = nfloat - n
            res = img[n, m] * (1 - w) + img[n + 1, m] * w

            drt[curr_h, curr_k] = res.sum() * params.dx
            tau += params.dtau

        p += params.dp


def _drt_sinc(img, drt, params):
    pass


class DRTParams:
    def __init__(
        self,
        k=101,
        h=101,
        dx=1,
        dy=1,
        x_min=0,
        y_min=0,
        p_start=-1,
        tau_start=0,
        dp=1 / 1024,
        dtau=0,
    ):
        self.k = k
        self.h = h
        self.dx = dx
        self.dy = dy
        self.x_min = x_min
        self.y_min = y_min
        self.p_start = p_start
        self.tau_start = tau_start
        self.dp = dp
        self.dtau = dy if dtau <= 0 else dtau


def drt(
    img,
    interpolation="linear",
    *,
    params=None,
    k=0,
    h=0,
    dx=1,
    dy=1,
    x_min=0,
    y_min=0,
    p_start=-1,
    tau_start=0,
    dp=0,
    dtau=0,
    drt_buffer=None,
):
    h = img.shape[0] if h <= 0 else h
    dp = dp if dp > 0 else dy / max(abs(x_min), x_min + dx * (img.shape[1] - 1))
    k = k if k > 0 else int(np.floor(2 / dp))
    if params is None:
        params = DRTParams(k, h, dx, dy, x_min, y_min, p_start, tau_start, dp, dtau)

    if drt_buffer is None:
        drt_buffer = np.zeros([params.h, params.k])

    if interpolation == "nearest":
        _drt_nearest(img, drt_buffer, params)
    elif interpolation == "linear":
        _drt_linear(img, drt_buffer, params)
    elif interpolation == "sinc":
        _drt_sinc(img, drt_buffer, params)
    else:
        raise ValueError(f"Unknown interpolation method: {interpolation}")

    return drt_buffer
