"""
Source module for end-to-end post-pretraining pipeline.

This file exists to make src a proper Python package, enabling imports like:
    from src.train import run_sft
    from src.inference import generate_response
"""

from src.train import *
from src.align import *
from src.inference import *
from src.data_processing import *
