"""Keep tests off CUDA. sftmill does not use the GPU."""

import os

if os.environ.get("SFTMILL_GPU_TESTS") != "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
