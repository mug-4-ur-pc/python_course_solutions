"""DRT Unit tests."""

import os
from typing import Generator


from click.testing import CliRunner
from discrete_radon_transform import console
import pytest


@pytest.fixture
def runner() -> CliRunner:
    """Fixture for invoking command-line interfaces."""
    return CliRunner()


@pytest.fixture
def input_img() -> str:
    """Fixture for input image filename."""
    return "tests/example.png"


@pytest.fixture
def tmp_output() -> str:
    """Fixture for tmp output filename."""
    return "tests/tmp_drt.png"


@pytest.fixture(autouse=True)
def manage_tmp_file(tmp_output: str) -> Generator:
    """Fixture to automatically delete output file."""
    yield
    if os.path.exists(tmp_output):
        os.remove(tmp_output)


def test_main_failed_without_input(runner: CliRunner, tmp_output: str) -> None:
    """It exits with non zero status code if input is not provided."""
    result = runner.invoke(console.main, ["-o", tmp_output])
    assert result.exit_code != 0


def test_main_succeeds(runner: CliRunner, input_img: str, tmp_output: str) -> None:
    """It exits with a status code of zero if input is provided."""
    result = runner.invoke(console.main, ["-s", input_img, "-o", tmp_output])
    assert result.exit_code == 0


def test_main_creates_output(
    runner: CliRunner, input_img: str, tmp_output: str
) -> None:
    """It creates an output file."""
    result = runner.invoke(console.main, ["-s", input_img, "-o", tmp_output])
    assert result.exit_code == 0
    assert os.path.exists(tmp_output)


@pytest.mark.parametrize("interp", ["nearest", "linear", "sinc"])
def test_main_interpolation(
    runner: CliRunner, input_img: str, tmp_output: str, interp: str
) -> None:
    """Different interpolations have no crashes."""
    result = runner.invoke(
        console.main, ["-s", input_img, "-i", interp, "-o", tmp_output]
    )
    assert result.exit_code == 0
    assert os.path.exists(tmp_output)


@pytest.mark.parametrize("threshold", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
def test_main_threshold(
    runner: CliRunner, input_img: str, tmp_output: str, threshold: float
) -> None:
    """Different interpolations have no crashes."""
    result = runner.invoke(
        console.main, ["-s", input_img, "-t", threshold, "-o", tmp_output]
    )
    assert result.exit_code == 0
    assert os.path.exists(tmp_output)
