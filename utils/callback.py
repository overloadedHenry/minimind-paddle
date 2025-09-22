from tqdm.auto import tqdm
class ProgressHandler:
    def __init__(self, total_steps, desc="Processing", enabled=True):
        """Tqdm wrapper that is no-op on non-main processes.

        enabled: whether to actually create and show the progress bar (only main process should set True)
        """
        self.enabled = enabled
        if enabled:
            self.pbar = tqdm(total=total_steps, desc=desc, leave=False, dynamic_ncols=True, ascii=True)
        else:
            self.pbar = None

    def update(self, n=1):
        if self.pbar is not None:
            self.pbar.update(n)

    def close(self):
        if self.pbar is not None:
            self.pbar.close()