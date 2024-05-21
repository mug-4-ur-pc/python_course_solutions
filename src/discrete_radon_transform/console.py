"""CLI for Discrete Radon Transform."""

import click
import cv2 as cv

from . import __version__
from .drt import find_lines


@click.command()
@click.option("-s", "--source", help="Input file")
@click.option("-o", "--output", default="drt.png", help="Output file")
@click.option(
    "-i",
    "--interpolation",
    default="linear",
    help="Interpolation method (see docs for details).",
)
@click.option(
    "-t", "--threshold", default=0.9, help="Threshold to extract lines from DRT space."
)
@click.version_option(version=__version__)
def main(source: str, output: str, interpolation: str, threshold: float) -> None:
    """I can tier all your enemies to pieces! Just ask for it!

    Args:
        source: Source image file.
        output: Output image file.
        interpolation: Interpolation method.
        threshold: Threshold to extract points from DRT space.
    """
    img = cv.imread(source)
    drt = find_lines(img, interpolation, threshold)
    cv.imwrite(output, drt)
