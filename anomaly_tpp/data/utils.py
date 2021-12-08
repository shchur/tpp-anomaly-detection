import torch


class DotDict:
    """Dictionary where elements can be accessed as dict.entry."""

    def __getitem__(self, key):
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def keys(self):
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def __iter__(self):
        for key in sorted(self.keys()):
            yield key, self[key]

    def __contains__(self, key):
        return key in self.keys()

    def apply_(self, func: callable):
        """Apply function to all attributes."""
        for key, value in self.items():
            self[key] = func(value)
        return self

    def to(self, device, non_blocking=False):
        """Move all tensors to the specified device."""

        def to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, non_blocking=non_blocking)
            else:
                return x

        return self.apply_(to_device)

    def cpu(self, non_blocking=False):
        """Move all tensors to CPU."""
        return self.to("cpu", non_blocking=non_blocking)

    def cuda(self, non_blocking=False):
        """Move all tensors to GPU."""
        return self.to("cuda", non_blocking=non_blocking)
