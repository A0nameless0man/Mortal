import sys
import logging
import warnings
import torch
import numpy as np

sys.stdin.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)8s %(filename)12s:%(lineno)-4s %(message)s",
    handlers=[logging.FileHandler("logs.txt"), logging.StreamHandler(sys.stdout)],
)

warnings.simplefilter("ignore")

# "The given NumPy array is not writeable"
dummy = np.array([])
dummy.setflags(write=False)
torch.as_tensor(dummy)

warnings.simplefilter("default")
