"""Audio metadata inspection helpers."""

from pathlib import Path
from typing import TypedDict, Union

import soundfile as sf


class AudioMetadata(TypedDict):
    """Header metadata returned by :func:`inspect_audio_metadata`."""

    path: str
    duration_seconds: float
    sample_rate_hz: int
    channels: int
    frames: int
    format: str
    subtype: str


def inspect_audio_metadata(file_path: Union[str, Path]) -> AudioMetadata:
    """Inspect an audio file's header without decoding its waveform.

    Uses ``soundfile.info`` to read metadata such as sample rate, channel
    count, frame count, duration, and encoding. This avoids allocating an
    array proportional to the audio file's duration.

    Args:
        file_path: Path to an audio file supported by libsndfile.

    Returns:
        A dictionary containing the resolved path and audio-header metadata.

    Raises:
        FileNotFoundError: If ``file_path`` is not an existing file.
        ValueError: If the file cannot be inspected or its audio metadata is
            invalid.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError, sf.SoundFileRuntimeError) as exc:
        raise ValueError(f"Could not inspect audio metadata for {path}: {exc}") from exc

    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError(f"Audio file has invalid or empty metadata: {path}")

    return {
        "path": str(path.resolve()),
        "duration_seconds": float(info.duration),
        "sample_rate_hz": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "format": info.format,
        "subtype": info.subtype,
    }
