"""Evaluation metrics for single- and multi-source audio separation.

The functions in this module accept one-dimensional NumPy arrays or Torch
tensors and return scalar scores in decibels. Torch inputs are detached and
converted to CPU NumPy arrays, so these metrics are intended for evaluation,
not for differentiable training losses.

For SDR, SIR, and SAR, the estimate is least-squares projected onto the target
reference and any supplied interference references. This decomposes the
estimate into target, interference, and artifact components. SI-SDR instead
projects only onto the target and measures the residual distortion, optionally
after removing the signals' means.
"""

from typing import Optional, Sequence, Union

import numpy as np
import torch


Signal = Union[np.ndarray, torch.Tensor]
_MAX_SCORE_DB = 100.0
_MIN_SCORE_DB = -100.0


def _as_signal(signal: Signal, name: str) -> np.ndarray:
    """Convert a signal to a finite, non-empty, one-dimensional float array."""
    if isinstance(signal, torch.Tensor):
        signal = signal.detach().cpu().numpy()
    result = np.asarray(signal, dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def _validate_pair(reference: Signal, estimation: Signal) -> tuple[np.ndarray, np.ndarray]:
    ref = _as_signal(reference, "reference")
    est = _as_signal(estimation, "estimation")
    if ref.shape != est.shape:
        raise ValueError(
            f"reference and estimation must have the same number of samples; "
            f"got {ref.size} and {est.size}."
        )
    return ref, est


def _interference_matrix(
    interference_references: Optional[Union[Signal, Sequence[Signal]]],
    sample_count: int,
) -> np.ndarray:
    """Return interference references as a matrix with one source per column."""
    if interference_references is None:
        return np.empty((sample_count, 0), dtype=np.float64)

    if isinstance(interference_references, torch.Tensor):
        raw = interference_references.detach().cpu().numpy()
    elif isinstance(interference_references, np.ndarray):
        raw = interference_references
    else:
        sources = list(interference_references)
        if not sources:
            return np.empty((sample_count, 0), dtype=np.float64)
        columns = [_as_signal(source, f"interference_references[{i}]") for i, source in enumerate(sources)]
        if any(source.size != sample_count for source in columns):
            raise ValueError("Each interference reference must match the sample count.")
        return np.column_stack(columns)

    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[np.newaxis, :]
    elif raw.ndim != 2:
        raise ValueError("interference_references must be a signal or a 2D source array.")
    if raw.shape[1] != sample_count:
        raise ValueError(
            "Each interference reference must match the sample count "
            f"({sample_count}); got {raw.shape[1]}."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("interference_references must contain only finite values.")
    return raw.T


def _energy_db(numerator: float, denominator: float, eps: float) -> float:
    """Convert an energy ratio to dB with finite, documented score limits."""
    if numerator <= eps:
        return _MIN_SCORE_DB
    if denominator <= eps:
        return _MAX_SCORE_DB
    return float(np.clip(10.0 * np.log10(numerator / denominator), _MIN_SCORE_DB, _MAX_SCORE_DB))


def _project_components(
    reference: Signal,
    estimation: Signal,
    interference_references: Optional[Union[Signal, Sequence[Signal]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return target, interference, and artifact projections of an estimate."""
    ref, est = _validate_pair(reference, estimation)
    other_refs = _interference_matrix(interference_references, ref.size)
    sources = np.column_stack((ref, other_refs))

    # Least squares gives the orthogonal projection onto the span of references,
    # including when references are linearly dependent.
    coefficients, _, _, _ = np.linalg.lstsq(sources, est, rcond=None)
    target = coefficients[0] * ref
    interference = other_refs @ coefficients[1:] if other_refs.shape[1] else np.zeros_like(est)
    artifact = est - target - interference
    return target, interference, artifact


def calculate_sisdr(
    reference: Signal,
    estimation: Signal,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> float:
    r"""Compute Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in dB.

    The estimate is decomposed into its least-squares projection onto the
    reference and the orthogonal residual:

    .. math::

       s_{\mathrm{target}} =
       \frac{\langle \hat{s}, s \rangle}{\|s\|^2}s,\qquad
       e_{\mathrm{noise}} = \hat{s} - s_{\mathrm{target}}

       \mathrm{SI\text{-}SDR} =
       10\log_{10}\frac{\|s_{\mathrm{target}}\|^2}
                              {\|e_{\mathrm{noise}}\|^2}.

    This score is invariant to a nonzero gain applied to the estimate. When
    ``zero_mean`` is true, the means are removed before the projection. Scores
    are limited to [-100, 100] dB; a denominator at or below ``eps`` receives
    100 dB, and a numerator at or below ``eps`` receives -100 dB.

    Args:
        reference: Clean target source.
        estimation: Estimated target source, with the same number of samples.
        eps: Energy threshold used for the zero-energy cases.
        zero_mean: Whether to remove each signal's mean before projection.

    Returns:
        SI-SDR in decibels.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    ref, est = _validate_pair(reference, estimation)
    if zero_mean:
        ref = ref - np.mean(ref)
        est = est - np.mean(est)

    reference_energy = float(np.dot(ref, ref))
    if reference_energy <= eps:
        return _MIN_SCORE_DB
    target = (float(np.dot(est, ref)) / reference_energy) * ref
    residual = est - target
    return _energy_db(float(np.dot(target, target)), float(np.dot(residual, residual)), eps)


def calculate_sdr(
    reference: Signal,
    estimation: Signal,
    eps: float = 1e-8,
    interference_references: Optional[Union[Signal, Sequence[Signal]]] = None,
) -> float:
    r"""Compute Signal-to-Distortion Ratio (SDR) in dB using source projections.

    The estimate is least-squares decomposed into its target projection,
    projections onto ``interference_references``, and a residual artifact
    orthogonal to their span. SDR is the target energy divided by the combined
    interference and artifact energy:

    .. math::

       \mathrm{SDR} =
       10\log_{10}\frac{\|s_{\mathrm{target}}\|^2}
          {\|e_{\mathrm{interference}}\|^2 + \|e_{\mathrm{artifact}}\|^2}.

    With no interference references, interference energy is zero and all
    non-target residual is counted as artifact. Scores are limited to
    [-100, 100] dB; zero denominators score 100 dB.

    Args:
        reference: Clean target source.
        estimation: Estimated source.
        eps: Energy threshold used for zero-energy cases.
        interference_references: Optional other clean sources. A single
            one-dimensional signal or a sequence of signals may be supplied.

    Returns:
        SDR in decibels.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    target, interference, artifact = _project_components(
        reference, estimation, interference_references
    )
    target_energy = float(np.dot(target, target))
    distortion_energy = float(np.dot(interference, interference) + np.dot(artifact, artifact))
    return _energy_db(target_energy, distortion_energy, eps)


def calculate_sir(
    reference: Signal,
    estimation: Signal,
    interference_references: Optional[Union[Signal, Sequence[Signal]]] = None,
    eps: float = 1e-8,
) -> float:
    r"""Compute Signal-to-Interference Ratio (SIR) in dB.

    After least-squares projection onto the target and interference references,
    SIR is the target projection energy divided by the summed energy of the
    interference projections:

    .. math::

       \mathrm{SIR} =
       10\log_{10}\frac{\|s_{\mathrm{target}}\|^2}
                              {\|e_{\mathrm{interference}}\|^2}.

    Artifacts are excluded from the denominator. If there are no interference
    references, or their projected interference energy is at or below ``eps``,
    SIR is reported as 100 dB. Scores are limited to [-100, 100] dB.

    Args:
        reference: Clean target source.
        estimation: Estimated source.
        interference_references: Other clean sources whose projections form
            the interference component; defaults to no interference sources.
        eps: Energy threshold used for zero-energy cases.

    Returns:
        SIR in decibels.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    target, interference, _ = _project_components(
        reference, estimation, interference_references
    )
    return _energy_db(
        float(np.dot(target, target)),
        float(np.dot(interference, interference)),
        eps,
    )


def calculate_sar(
    reference: Signal,
    estimation: Signal,
    interference_references: Optional[Union[Signal, Sequence[Signal]]] = None,
    eps: float = 1e-8,
) -> float:
    r"""Compute Sources-to-Artifacts Ratio (SAR) in dB.

    SAR compares the combined target and interference energy in the projected
    estimate with the artifact residual, which is orthogonal to the span of all
    supplied reference sources:

    .. math::

       \mathrm{SAR} =
       10\log_{10}
       \frac{\|s_{\mathrm{target}}\|^2 + \|e_{\mathrm{interference}}\|^2}
            {\|e_{\mathrm{artifact}}\|^2}.

    If no interference references are provided, the numerator contains only
    target energy. Scores are limited to [-100, 100] dB; an artifact energy at
    or below ``eps`` scores 100 dB.

    Args:
        reference: Clean target source.
        estimation: Estimated source.
        interference_references: Optional other clean sources.
        eps: Energy threshold used for zero-energy cases.

    Returns:
        SAR in decibels.
    """
    if eps <= 0:
        raise ValueError("eps must be positive.")
    target, interference, artifact = _project_components(
        reference, estimation, interference_references
    )
    desired_energy = float(np.dot(target, target) + np.dot(interference, interference))
    artifact_energy = float(np.dot(artifact, artifact))
    return _energy_db(desired_energy, artifact_energy, eps)
