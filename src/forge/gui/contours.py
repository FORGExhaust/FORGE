"""Helpers for extracting contour line data from matplotlib ContourSets."""

from __future__ import annotations

from typing import Iterator

import numpy as np
from matplotlib.path import Path as MplPath


def _segments_from_path(path) -> Iterator[np.ndarray]:
    """Yield open polyline segments from a single matplotlib Path."""
    verts = path.vertices
    if len(verts) == 0:
        return

    codes = path.codes
    if codes is None:
        yield verts
        return

    starts = np.where(codes == MplPath.MOVETO)[0]
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(verts)
        segment = verts[start:end]
        if len(segment) >= 2:
            yield segment


def iter_contour_segments(contour_set) -> Iterator[np.ndarray]:
    """Yield vertex arrays for each contour line segment in *contour_set*.

    Matplotlib 3.8+ removed ``ContourSet.collections``; older versions used one
    ``PathCollection`` per level.  This helper supports both layouts.
    """
    collections = getattr(contour_set, "collections", None)
    if collections is not None:
        for collection in collections:
            for path in collection.get_paths():
                yield from _segments_from_path(path)
        return

    for path in contour_set.get_paths():
        yield from _segments_from_path(path)


def contour_xy_lists(contour_set) -> tuple[list, list]:
    """Return ``(xs_all, ys_all)`` line lists for Bokeh ``multi_line``."""
    xs_all, ys_all = [], []
    for verts in iter_contour_segments(contour_set):
        xs_all.append(verts[:, 0].tolist())
        ys_all.append(verts[:, 1].tolist())
    return xs_all, ys_all


def colored_contour_xy_lists(contour_set, cmap, norm) -> tuple[list, list, list]:
    """Return ``(xs_all, ys_all, colors_all)`` for per-level coloured contours."""
    import matplotlib.colors

    xs_all, ys_all, colors_all = [], [], []

    collections = getattr(contour_set, "collections", None)
    if collections is not None:
        for level_val, collection in zip(contour_set.levels, collections):
            color = matplotlib.colors.to_hex(cmap(norm(level_val)))
            for path in collection.get_paths():
                for segment in _segments_from_path(path):
                    xs_all.append(segment[:, 0].tolist())
                    ys_all.append(segment[:, 1].tolist())
                    colors_all.append(color)
        return xs_all, ys_all, colors_all

    for level_val, path in zip(contour_set.levels, contour_set.get_paths()):
        color = matplotlib.colors.to_hex(cmap(norm(level_val)))
        for segment in _segments_from_path(path):
            xs_all.append(segment[:, 0].tolist())
            ys_all.append(segment[:, 1].tolist())
            colors_all.append(color)
    return xs_all, ys_all, colors_all
