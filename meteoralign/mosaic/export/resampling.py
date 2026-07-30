"""重投影图像的像素重采样方法。"""

from __future__ import annotations

from typing import Literal

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - 由上层导出可用性检查给出友好错误。
    cv2 = None


MosaicResamplingMethod = Literal["bilinear", "bicubic", "lanczos3"]
MOSAIC_RESAMPLING_METHODS: tuple[MosaicResamplingMethod, ...] = (
    "bilinear",
    "bicubic",
    "lanczos3",
)
MOSAIC_DEFAULT_RESAMPLING_METHOD: MosaicResamplingMethod = "bilinear"
_LANCZOS3_RADIUS = 3
_LANCZOS3_CHUNK_PIXELS = 131_072


def remap_rgb_image(
    source_rgb: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    *,
    method: MosaicResamplingMethod = MOSAIC_DEFAULT_RESAMPLING_METHOD,
) -> np.ndarray:
    """按指定方法把 RGB 源图采样到目标坐标。

    OpenCV 原生提供双线性和双三次 remap，但只提供 Lanczos4。为了让界面
    中的 Lanczos3 名称与实际 3-lobe 核一致，这里单独实现可分离 Lanczos3。
    """

    if method not in MOSAIC_RESAMPLING_METHODS:
        raise ValueError(f"不支持的重投影插值方式：{method!r}")
    source = np.ascontiguousarray(source_rgb)
    x_coordinates = np.asarray(map_x, dtype=np.float32)
    y_coordinates = np.asarray(map_y, dtype=np.float32)
    if x_coordinates.shape != y_coordinates.shape:
        raise ValueError("重投影横纵坐标 map 尺寸不一致。")
    if method == "lanczos3":
        return _remap_lanczos3(source, x_coordinates, y_coordinates)
    if cv2 is None:
        raise RuntimeError("当前环境缺少 OpenCV，无法执行重投影导出。")
    interpolation = cv2.INTER_LINEAR if method == "bilinear" else cv2.INTER_CUBIC
    return cv2.remap(
        source,
        x_coordinates,
        y_coordinates,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def _remap_lanczos3(
    source_rgb: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
) -> np.ndarray:
    """使用半径为 3 的标准可分离 Lanczos 核执行任意坐标重采样。"""

    if source_rgb.ndim != 3:
        raise ValueError(f"Lanczos3 重采样需要 HxWxC 源图，实际为 {source_rgb.shape}。")
    output_shape = (*map_x.shape, int(source_rgb.shape[2]))
    output = np.zeros(output_shape, dtype=source_rgb.dtype)
    flat_output = output.reshape(-1, output_shape[-1])
    flat_x = map_x.reshape(-1)
    flat_y = map_y.reshape(-1)
    source_height, source_width = source_rgb.shape[:2]
    offsets = np.arange(
        -_LANCZOS3_RADIUS + 1,
        _LANCZOS3_RADIUS + 1,
        dtype=np.int64,
    )
    dtype_info = np.iinfo(source_rgb.dtype)

    for start in range(0, flat_x.size, _LANCZOS3_CHUNK_PIXELS):
        end = min(flat_x.size, start + _LANCZOS3_CHUNK_PIXELS)
        x_values = flat_x[start:end].astype(np.float64, copy=False)
        y_values = flat_y[start:end].astype(np.float64, copy=False)
        finite = np.isfinite(x_values) & np.isfinite(y_values)
        safe_x = np.where(finite, x_values, 0.0)
        safe_y = np.where(finite, y_values, 0.0)
        x_indices = np.floor(safe_x).astype(np.int64)[:, None] + offsets
        y_indices = np.floor(safe_y).astype(np.int64)[:, None] + offsets
        x_weights = _normalized_lanczos3_weights(safe_x[:, None] - x_indices)
        y_weights = _normalized_lanczos3_weights(safe_y[:, None] - y_indices)
        accumulated = np.zeros((end - start, output_shape[-1]), dtype=np.float64)

        for y_tap in range(offsets.size):
            source_y = y_indices[:, y_tap]
            valid_y = (source_y >= 0) & (source_y < source_height)
            clipped_y = np.clip(source_y, 0, source_height - 1)
            for x_tap in range(offsets.size):
                source_x = x_indices[:, x_tap]
                in_bounds = finite & valid_y & (source_x >= 0) & (source_x < source_width)
                if not np.any(in_bounds):
                    continue
                clipped_x = np.clip(source_x, 0, source_width - 1)
                weights = y_weights[:, y_tap] * x_weights[:, x_tap] * in_bounds
                accumulated += (
                    source_rgb[clipped_y, clipped_x].astype(np.float64, copy=False)
                    * weights[:, None]
                )

        flat_output[start:end] = np.clip(
            np.rint(accumulated),
            dtype_info.min,
            dtype_info.max,
        ).astype(source_rgb.dtype)

    return output


def _normalized_lanczos3_weights(distances: np.ndarray) -> np.ndarray:
    within_kernel = np.abs(distances) < float(_LANCZOS3_RADIUS)
    weights = np.where(
        within_kernel,
        np.sinc(distances) * np.sinc(distances / float(_LANCZOS3_RADIUS)),
        0.0,
    )
    totals = np.sum(weights, axis=1, keepdims=True)
    return np.divide(
        weights,
        totals,
        out=np.zeros_like(weights),
        where=np.abs(totals) > 1.0e-12,
    )
