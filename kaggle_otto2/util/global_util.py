import os
import random
import time
from typing import Tuple

import numpy as np
import psutil
import torch
import torch.backends.cudnn
import torch.cuda


class GlobalUtil:
    @staticmethod
    def seed_everything(seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if GlobalUtil.torch_cuda_is_usable():
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    @staticmethod
    def get_metric() -> Tuple[float, float, float]:
        t = time.time()
        p = psutil.Process(os.getpid())
        m: float = p.memory_info()[0] / 2.0**30
        per: float = psutil.virtual_memory().percent
        return t, m, per

    @staticmethod
    def get_torch_device() -> str:
        return "cuda" if GlobalUtil.torch_cuda_is_usable() else "cpu"

    @staticmethod
    def torch_cuda_is_usable() -> bool:
        if os.environ.get("OTTO_FORCE_CPU", "").lower() in {"1", "true", "yes"}:
            return False
        if not torch.cuda.is_available():
            return False

        try:
            major, minor = torch.cuda.get_device_capability()
            arch = f"sm_{major}{minor}"
            supported_arches = set(torch.cuda.get_arch_list())
            if supported_arches and arch not in supported_arches:
                return False
        except Exception:
            return False

        return True
