from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


ROOT = Path(__file__).resolve().parent

ext_modules = [
    Pybind11Extension(
        "training.tpu.data.cpp_ext._fast_fen",
        [str(ROOT / "fast_fen.cpp")],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]


setup(
    name="mythos-fast-fen",
    version="0.1.0",
    description="Fast FEN parser for Mythos TPU training",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
