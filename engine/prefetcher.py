import torch


def _to_device(obj, device, non_blocking=True):
    """Recursively move tensors (including inside lists/dicts of dicts, as
    produced by collate_fn's (images, list_of_target_dicts)) onto device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: _to_device(v, device, non_blocking) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device, non_blocking) for v in obj)
    return obj


class CUDAPrefetcher:
    """Overlaps host->device transfer of the *next* batch with GPU compute on
    the *current* batch, using a side CUDA stream.

    Why this matters on Kaggle: sessions typically give only 2-4 CPU cores,
    so image decode/augment in the DataLoader workers is often slower than a
    single forward+backward step. Without overlap, the GPU sits idle every
    step waiting on `.to(device)` for a batch that a worker only just
    finished collating. Prefetching the next batch's H2D copy during the
    current step's compute keeps the GPU fed continuously instead of
    alternating between "GPU busy" and "GPU idle waiting on CPU/PCIe".

    Falls back to plain (blocking, non-overlapped) transfers on CPU, since
    CUDA streams don't apply there.
    """

    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.is_cuda = (device.type == 'cuda')
        self.stream = torch.cuda.Stream() if self.is_cuda else None
        self._iter = None

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        self._iter = iter(self.loader)
        self._preload()
        return self

    def _preload(self):
        try:
            self.next_samples, self.next_targets = next(self._iter)
        except StopIteration:
            self.next_samples, self.next_targets = None, None
            return

        if not self.is_cuda:
            return

        # Copy on a side stream so it doesn't block (or get blocked by) the
        # default stream, which is busy running the previous step's compute.
        with torch.cuda.stream(self.stream):
            self.next_samples = self.next_samples.to(self.device, non_blocking=True)
            self.next_targets = _to_device(self.next_targets, self.device, non_blocking=True)

    def __next__(self):
        if self.next_samples is None:
            raise StopIteration

        if self.is_cuda:
            # Make sure the default stream waits for the prefetch copy to
            # finish before compute on these tensors starts.
            torch.cuda.current_stream().wait_stream(self.stream)
            samples, targets = self.next_samples, self.next_targets
            # Record the current stream on each tensor so the caching
            # allocator doesn't free/reuse the memory until compute is done
            # (required when a tensor was allocated on a different stream).
            samples.record_stream(torch.cuda.current_stream())
            for t in targets:
                for v in t.values():
                    if torch.is_tensor(v):
                        v.record_stream(torch.cuda.current_stream())
        else:
            samples, targets = self.next_samples, self.next_targets

        self._preload()
        return samples, targets
