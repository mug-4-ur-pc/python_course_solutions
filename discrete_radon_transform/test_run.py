from drt import drt
import numpy as np

import imageio.v3 as iio
import plotly.express as px


def get_drt_graph(img, interpolation):
    d, p = drt(img, interpolation=interpolation)

    fig = px.imshow(
        d,
        labels=dict(x="p", y="tau"),
        y=np.arange(p.tau_min, p.tau_max + p.dtau, p.dtau),
        x=np.arange(p.p_min, p.p_max + p.dp, p.dp),
        color_continuous_scale="gray",
        aspect="auto",
    )
    fig["layout"]["yaxis"].update(autorange=True)
    fig.write_image("img/drt_" + interpolation + ".png")


if __name__ == "__main__":
    img = iio.imread("img/example.png")
    for i in ["nearest", "linear", "sinc"]:
        get_drt_graph(img, i)
