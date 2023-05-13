import math
from typing import Callable, List
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

class LinearWarmUpCosineAnnealingLR(LambdaLR):
    def __init__(self, optimizer, *, peak, final, warm_up_steps, max_steps, init=1e-8, offset=0, **kwargs):
        assert peak >= final >= init >= 0
        assert max_steps >= warm_up_steps
        self.init = init
        self.peak = peak
        self.final = final
        self.warm_up_steps = warm_up_steps
        self.max_steps = max_steps
        self.offset = offset
        kwargs['optimizer'] = optimizer
        kwargs['lr_lambda'] = self._step_inner
        super().__init__(**kwargs)

    def _step_inner(self, steps):
        steps += self.offset
        if self.warm_up_steps > 0 and steps < self.warm_up_steps:
            return self.init + (self.peak - self.init) / self.warm_up_steps * steps
        if steps < self.max_steps:
            cos_steps = steps - self.warm_up_steps
            cos_max_steps = self.max_steps - self.warm_up_steps
            return self.final + 0.5 * (self.peak - self.final) * (1 + np.cos(cos_steps / cos_max_steps * np.pi))
        return self.final

def step_scheduler(step_size: int, gamma: float, **kargs) -> Callable[[int], float]:
    def calc(step: int) -> float:
        return gamma ** (step // step_size)

    return calc

def exp_step_scheduler(
    step_size: int, gamma: float, **kargs
) -> Callable[[int], float]:
    def calc(step: int) -> float:
        return gamma ** math.log(step // step_size, 2)

    return calc

def stage_scheduler_step(
    left=0, right=-1, type="step", **kargs
) -> Callable[[int], float]:
    match type:
        case "step":
            internal_calc = step_scheduler(**kargs)
        case "exp_step":
            internal_calc = exp_step_scheduler(**kargs)
        case _:
            raise IndexError()

    def calc(step: int):
        if step <= left:
            return 1
        if left != -1 and step > right:
            step = right
        step -= left
        return internal_calc(step)

    return calc

def stage_scheduler(stages: List[dict]) -> Callable[[int], float]:
    funcs = [stage_scheduler_step(**args) for args in stages]

    def calc(step: int):
        gamma = 1.0
        for f in funcs:
            gamma *= f(step)
        return gamma

    return calc