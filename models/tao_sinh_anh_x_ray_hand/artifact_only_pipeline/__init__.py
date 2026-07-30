"""Pixel-preserving removal of non-anatomical X-ray labels."""

from .pipeline import ArtifactOnlyPipeline, ProcessResult

__all__ = ["ArtifactOnlyPipeline", "ProcessResult"]

