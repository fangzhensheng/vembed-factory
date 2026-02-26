import torch
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    """Save and restore RNG state for deterministic recomputation.

    Uses fork_rng for proper scoping across all GPU devices.
    Handles None tensors with fallback to all available devices.
    """

    def __init__(self, *tensors):
        """Save current RNG state for later restoration."""
        self.fwd_cpu_state = torch.get_rng_state()

        try:
            self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)
        except Exception:
            if torch.cuda.is_available():
                self.fwd_gpu_devices = list(range(torch.cuda.device_count()))
                self.fwd_gpu_states = [
                    torch.cuda.get_rng_state(device=i) for i in self.fwd_gpu_devices
                ]
            else:
                self.fwd_gpu_devices = []
                self.fwd_gpu_states = []

        self._fork = None

    def __enter__(self):
        """Restore saved RNG state using fork_rng for proper scoping."""
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()

        torch.set_rng_state(self.fwd_cpu_state)
        if self.fwd_gpu_devices and self.fwd_gpu_states:
            set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore previous RNG state when exiting scope."""
        if self._fork is not None:
            self._fork.__exit__(exc_type, exc_val, exc_tb)
            self._fork = None
