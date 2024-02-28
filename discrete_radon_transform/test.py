from drt import drt
import imageio as iio
import matplotlib.pyplot as plt

img = iio.imread("drt.png")
img = img.mean(axis=2).astype(int)

d = drt(img)

fig, ax = plt.subplots()
im = ax.imshow(d, cmap="gray", vmin=d.min(), vmax=d.max())
cbar = fig.colorbar(im, label="brightness")
fig.savefig("fig.png")
