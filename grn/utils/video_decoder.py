from abc import ABC, abstractmethod
import io
import math
import numpy as np
from typing import Optional, TypeVar, Union
import collections

try:
    import decord
except ImportError:
    _HAS_DECORD = False
else:
    _HAS_DECORD = True

if _HAS_DECORD:
    decord.bridge.set_bridge('native')

DecordDevice = TypeVar("DecordDevice")

# https://github.com/dmlc/decord/issues/208#issuecomment-1157632702
if _HAS_DECORD:
    class VideoReaderWrapper(decord.VideoReader):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.seek(0)

        def __getitem__(self, key):
            frames = super().__getitem__(key)
            self.seek(0)
            return frames
else:
    class VideoReaderWrapper:
        """Placeholder that keeps this optional decoder module importable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "decord is required to use EncodedVideoDecord decoder. Please "
                "install with 'pip install decord' for CPU-only version and refer to "
                "'https://github.com/dmlc/decord' for GPU-supported version"
            )


class Video(ABC):
    """
    Video provides an interface to access clips from a video container.
    """

    @abstractmethod
    def __init__(
        self,
        file: Union[str, io.IOBase],
        video_name: Optional[str] = None,
        decode_audio: bool = True,
    ) -> None:
        """
        Args:
            file (BinaryIO): a file-like object (e.g. io.BytesIO or io.StringIO) that
                contains the encoded video.
        """
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """
        Returns:
            duration of the video in seconds
        """
        pass

    @abstractmethod
    def get_clip(
        self, start_sec: float, end_sec: float, num_samples: int
    ):
        """
        Retrieves frames from the internal video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            video_data_dictonary: A dictionary mapping strings to tensor of the clip's
                underlying data.

        """
        pass

    def close(self):
        pass


class EncodedVideoDecord(Video):
    """

    Accessing clips from an encoded video using Decord video reading API
    as the decoding backend. For more details, please refer to -
    `Decord <https://github.com/dmlc/decord>`
    """

    def __init__(
        self,
        file: Union[str, io.IOBase],
        video_name: Optional[str] = None,
        width: int = -1,
        height: int = -1,
        num_threads: int = 0,
        fault_tol: int = -1,
    ) -> None:
        """
        Args:
            file str: file path.
            video_name (str): An optional name assigned to the video.
            decode_audio (bool): If disabled, audio is not decoded.
            sample_rate: int, default is -1
                Desired output sample rate of the audio, unchanged if `-1` is specified.
            mono: bool, default is True
                Desired output channel layout of the audio. `True` is mono layout. `False`
                is unchanged.
            width : int, default is -1
                Desired output width of the video, unchanged if `-1` is specified.
            height : int, default is -1
                Desired output height of the video, unchanged if `-1` is specified.
            num_threads : int, default is 0
                Number of decoding thread, auto if `0` is specified.
            fault_tol : int, default is -1
                The threshold of corupted and recovered frames. This is to prevent silent fault
                tolerance when for example 50% frames of a video cannot be decoded and duplicate
                frames are returned. You may find the fault tolerant feature sweet in many
                cases, but not for training models. Say `N = # recovered frames`
                If `fault_tol` < 0, nothing will happen.
                If 0 < `fault_tol` < 1.0, if N > `fault_tol * len(video)`,
                raise `DECORDLimitReachedError`.
                If 1 < `fault_tol`, if N > `fault_tol`, raise `DECORDLimitReachedError`.
        """
        self._video_name = video_name
        if not _HAS_DECORD:
            raise ImportError(
                "decord is required to use EncodedVideoDecord decoder. Please "
                "install with 'pip install decord' for CPU-only version and refer to"
                "'https://github.com/dmlc/decord' for GPU-supported version"
            )
        try:
            self._av_reader = VideoReaderWrapper(
                uri=file,
                ctx=decord.cpu(0),
                width=width,
                height=height,
                num_threads=num_threads,
                fault_tol=fault_tol,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to open video {video_name} with Decord. {e}")

        self._fps = self._av_reader.get_avg_fps()
        self._duration = float(len(self._av_reader)) / float(self._fps)

    @property
    def name(self) -> Optional[str]:
        """
        Returns:
            name: the name of the stored video if set.
        """
        return self._video_name

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def close(self):
        if self._av_reader is not None:
            del self._av_reader
            self._av_reader = None

    def get_clip(
        self, start_sec: float, end_sec: float, num_samples: int
    ):
        """
        Retrieves frames from the encoded video at the specified start and end times
        in seconds (the video always starts at 0 seconds).

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds
        Returns:
            clip_data:
                A dictionary mapping the entries at "video" and "audio" to a tensors.

                "video": A tensor of the clip's RGB frames with shape:
                (channel, time, height, width). The frames are of type torch.float32 and
                in the range [0 - 255].

            Returns None if no video or audio found within time range.

        """
        if start_sec > end_sec or start_sec > self._duration:
            raise RuntimeError(
                f"Incorrect time window for Decord decoding for video: {self._video_name}."
            )

        start_idx = math.ceil(self._fps * start_sec)
        end_idx = math.ceil(self._fps * end_sec)
        end_idx = min(end_idx, len(self._av_reader))
        # frame_idxs = list(range(start_idx, end_idx))

        frame_idxs = np.linspace(start_idx, end_idx - 1, num_samples, dtype=int)

        try:
            outputs = self._av_reader.get_batch(frame_idxs)
            return outputs.asnumpy(), frame_idxs - frame_idxs[0]
        except Exception as e:
            print(f"Failed to decode video with Decord: {self._video_name}. {e}")
            raise e

try:
    import cv2
except ImportError:
    print(f"ERR: import cv2 failed, install cv2 by 'pip install opencv-python'")

class EncodedVideoOpencv():
    def __init__(
        self,
        file: Union[str, io.IOBase],
        video_name: Optional[str] = None,
        width: int = -1,
        height: int = -1,
        num_threads: int = 0,
        fault_tol: int = -1,
    ) -> None:
        """
        Args:
            file str: file path.
            video_name (str): An optional name assigned to the video.
            width : Not support yet.
            height : Not support yet.
            num_threads : Not support yet.
            fault_tol : Not support yet.
        """

        self._video_name = video_name
        self.cap = cv2.VideoCapture(file)
        self._fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._vlen = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = float(self._vlen) / float(self._fps)

    @property
    def name(self) -> Optional[str]:
        """
        Returns:
            name: the name of the stored video if set.
        """
        return self._video_name

    @property
    def duration(self) -> float:
        """
        Returns:
            duration: the video's duration/end-time in seconds.
        """
        return self._duration

    def __del__(self):
        self.close()

    def close(self):
        self.cap.release()

    def get_clip(
        self, start_sec: float, end_sec: float, num_samples: int
    ):
        if start_sec > end_sec or start_sec > self._duration:
            raise RuntimeError(
                f"Incorrect time window for Decord decoding for video: {self._video_name}."
            )
        start_idx = math.ceil(self._fps * start_sec)
        end_idx = math.ceil(self._fps * end_sec)
        end_idx = min(end_idx, self._vlen)
        frame_idxs = np.linspace(start_idx, end_idx - 1, num_samples, dtype=int)
        frame_idx2freq = collections.defaultdict(int)
        for frame_idx in frame_idxs:
            frame_idx2freq[frame_idx] += 1
        try:
            frames = []
            for i in range(self._vlen):
                if i > frame_idxs[-1]:
                    break
                ret, frame = self.cap.read()
                if i in frame_idx2freq:
                    frames.extend([frame] * frame_idx2freq[i])
            frames = np.array(frames).astype(np.uint8) # BGR type
            assert len(frames) == num_samples
            return frames, frame_idxs - frame_idxs[0]
        except Exception as e:
            print(f"Failed to decode video with opencv: {self._video_name}. {e}")
            raise e
