from drt import drt
import numpy as np

import imageio.v3 as iio
import plotly.express as px

img = iio.imread("img/drt.png")
img = img.mean(axis=2).astype(int)

d = drt(
    img,
    interpolation="nearest",
    p_start=-img.shape[0],
    p_amount=img.shape[0] * 6,
    dp=0.5,
    tau_start=-1.5,
    tau_amount=6 * img.shape[1],
    dtau=0.5 / img.shape[1],
)

fig = px.imshow(
    d,
    labels=dict(x="p", y="tau"),
    x=np.linspace(-img.shape[0], img.shape[0] * 2, img.shape[0] * 6),
    y=np.linspace(-1.5, 1.5, 6 * img.shape[1]),
    color_continuous_scale="gray",
    aspect="auto",
)
fig["layout"]["yaxis"].update(autorange=True)
fig.write_image("img/fig.png")
