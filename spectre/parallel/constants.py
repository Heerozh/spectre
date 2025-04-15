import torch


class TensorConstant:
    def __init__(self, device):
        self.device = device
        self.linspace_cache = {}
        self.r_linspace_cache = {}
        self.arange_cache = {}

    def linspace(self, size, dtype):
        if dtype in self.linspace_cache:
            w = self.linspace_cache[dtype]
            if size <= len(w):
                return w[:size]

        self.linspace_cache[dtype] = new = torch.linspace(
            0.0, 0.9, size, dtype=dtype, device=self.device)
        return new

    def r_linspace(self, size, dtype):
        if dtype in self.r_linspace_cache:
            w = self.r_linspace_cache[dtype]
            if size <= len(w):
                return w[:size]

        self.r_linspace_cache[dtype] = new = torch.linspace(
            0.9, 0.0, size, dtype=dtype, device=self.device)
        return new

    def arange(self, size, dtype):
        if dtype in self.arange_cache:
            w = self.arange_cache[dtype]
            if size <= len(w):
                return w[:size]

        self.arange_cache[dtype] = new = torch.arange(size, dtype=dtype, device=self.device)
        return new


class DeviceConstant:
    constants = {}

    @classmethod
    def clean(cls):
        cls.constants = {}

    @classmethod
    def get(cls, device):
        if device in cls.constants:
            return cls.constants[device]

        cls.constants[device] = new = TensorConstant(device)
        return new





