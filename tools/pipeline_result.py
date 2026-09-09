import os


class PipelineResult:
    """Generated media and the cleanup handle for any returned video file."""

    def __init__(self, images=None, videos=None, cleanup_paths=()):
        self.images = images or []
        self.videos = videos or []
        self._cleanup_paths = tuple(cleanup_paths)

    def cleanup(self):
        """Remove temporary video outputs created by the pipeline."""
        for path in self._cleanup_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._cleanup_paths = ()
