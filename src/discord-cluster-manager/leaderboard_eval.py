########
# Evaluation scripts to run for leaderboard results
########

from pathlib import Path

py_eval = Path.read_text(Path(__file__).parent / "eval.py")
cu_eval = Path.read_text(Path(__file__).parent / "eval.cu")

nvidia_requirements = """
numpy
torch
setuptools
ninja
triton
"""

amd_requirements = """
--index-url https://download.pytorch.org/whl/nightly/rocm6.2
pytorch-triton-rocm==3.1.0+cf34004b8a
torch==2.6.0.dev20241023+rocm6.2
"""
