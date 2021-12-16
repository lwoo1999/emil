from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import integrate

@dataclass
class RSR:
    xs: List[float]
    ys: List[float]

    def normalise(self):
        total = integrate.trapz(self.ys, x=self.xs)
        self.ys /= total

    def redshift(self, z):
        return RSR(self.xs / (1+z), self.ys * (1+z))