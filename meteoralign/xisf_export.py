from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import numpy as np
from xisf import XISF

from .frame_astrometry import FrameAstrometricModel
from .native_image import load_native_image_array


XISF_EXPORT_MODES: dict[str, tuple[int, str]] = {
    "fast": (25, "PixInsight 打开较快，约 634 个控制点（推荐）"),
    "lite": (17, "PixInsight 打开最快，约 298 个控制点"),
    "high": (45, "PixInsight 打开较慢，约 2034 个控制点"),
}

PIXINSIGHT_ASTROMETRIC_PROPERTIES = (
    "PCL:AstrometricSolution:ProjectionSystem",
    "PCL:AstrometricSolution:ReferenceCelestialCoordinates",
    "PCL:AstrometricSolution:ReferenceImageCoordinates",
    "PCL:AstrometricSolution:LinearTransformationMatrix",
    "PCL:AstrometricSolution:SplineWorldTransformation:ControlPoints:Image",
    "PCL:AstrometricSolution:SplineWorldTransformation:ControlPoints:World",
)


@dataclass(frozen=True)
class XisfValidationResult:
    pair_count: int
    validated_count: int
    median_error_arcsec: float


@dataclass(frozen=True)
class XisfExportResult:
    output_path: Path
    file_size_bytes: int
    codec: str | None
    control_point_count: int
    max_zea_radius_deg: float
    validation: XisfValidationResult


def angular_errors_arcsec(reference: np.ndarray, measured: np.ndarray) -> np.ndarray:
    """返回两组 ICRS 坐标之间的球面角距离（角秒）。"""

    ra1, dec1 = np.deg2rad(reference[:, 0]), np.deg2rad(reference[:, 1])
    ra2, dec2 = np.deg2rad(measured[:, 0]), np.deg2rad(measured[:, 1])
    cosine = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) * 3600.0


def validate_star_pairs(
    model: FrameAstrometricModel,
    pairs: Sequence[Mapping[str, object]],
) -> XisfValidationResult:
    """在写文件前确认当前匹配星与待嵌入的天文模型一致。"""

    if len(pairs) < 4:
        raise ValueError("有效星点匹配不足，至少需要 4 对才能导出 XISF。")
    try:
        pixels = np.asarray(
            [[float(pair["image_x_px"]), float(pair["image_y_px"])] for pair in pairs],
            dtype=np.float64,
        )
        expected = np.asarray(
            [[float(pair["ra_deg"]), float(pair["dec_deg"])] for pair in pairs],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("星点匹配记录缺少有效的像素或 ICRS 坐标。") from exc

    measured = model.pixel_to_sky_points(pixels)
    valid = np.all(np.isfinite(expected), axis=1) & np.all(np.isfinite(measured), axis=1)
    validated_count = int(np.count_nonzero(valid))
    if validated_count < 4:
        raise ValueError("当前天文模型无法反解足够的匹配星点。")
    median_error = float(np.median(angular_errors_arcsec(expected[valid], measured[valid])))
    if median_error > 3600.0:
        raise ValueError(
            f"星点匹配与天文模型可能不配套：反解中位误差 {median_error:.1f} arcsec。"
        )
    return XisfValidationResult(
        pair_count=len(pairs),
        validated_count=validated_count,
        median_error_arcsec=median_error,
    )


def _zea_world(radec: np.ndarray, center_radec: np.ndarray) -> np.ndarray:
    """将 ICRS 坐标投影到 PixInsight 使用的 ZEA 世界平面。"""

    ra, dec = np.deg2rad(radec[:, 0]), np.deg2rad(radec[:, 1])
    ra0, dec0 = np.deg2rad(center_radec)
    vectors = np.column_stack(
        (np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec))
    )
    center = np.array(
        [np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)]
    )
    east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
    north = np.array(
        [-np.sin(dec0) * np.cos(ra0), -np.sin(dec0) * np.sin(ra0), np.cos(dec0)]
    )
    theta = np.arccos(np.clip(vectors @ center, -1.0, 1.0))
    radius = (360.0 / np.pi) * np.sin(theta / 2.0)
    scale = np.divide(
        radius,
        np.sin(theta),
        out=np.ones_like(radius),
        where=np.abs(np.sin(theta)) > 1e-12,
    )
    return np.column_stack(((vectors @ east) * scale, (vectors @ north) * scale))


def _extended_equisolid_icrs(
    model: FrameAstrometricModel,
    pixels: np.ndarray,
) -> np.ndarray:
    """把等立体角鱼眼圆外区域平滑延拓，确保 PixInsight 能验证矩形四角。"""

    profile = model.camera_calibration_profile
    if profile.base_projection_type != "fisheye_equisolid":
        return model.pixel_to_sky_points(pixels)

    points = np.asarray(pixels, dtype=np.float64)
    profile_pixels = model.frame_local_residual.apply_inverse(points)
    raw, distortion_valid = profile._uncorrect_global_distortion(profile_pixels)
    # 圆外没有真实天空；反畸变失败时只将输入作为构造安全延拓的初值。
    raw[~distortion_valid] = profile_pixels[~distortion_valid]
    valid = np.all(np.isfinite(raw), axis=1)
    x = (raw[:, 0] - profile.principal_point_x_px) / profile.scale_x_px
    y = (profile.principal_point_y_px - raw[:, 1]) / profile.scale_y_px
    radius = np.hypot(x, y)
    preserve_radius = 2.0 * np.sin(np.deg2rad(160.0) / 2.0)
    max_radius = 2.0 * np.sin(np.deg2rad(170.0) / 2.0)
    transition = max(max_radius - preserve_radius, 1e-6)
    outside = np.maximum(radius - preserve_radius, 0.0)
    extended = preserve_radius + transition * (1.0 - np.exp(-outside / transition))
    safe_radius = np.where(radius <= preserve_radius, radius, extended)
    theta = 2.0 * np.arcsin(np.clip(safe_radius / 2.0, 0.0, 1.0))
    factor = np.divide(
        np.sin(theta),
        radius,
        out=np.ones_like(radius),
        where=radius > 1e-12,
    )
    camera = np.column_stack((x * factor, y * factor, np.cos(theta)))
    camera[~valid] = np.nan
    icrs = model.frame_pose.camera_vectors_to_icrs(camera)
    norm = np.linalg.norm(icrs, axis=1)
    finite = valid & np.all(np.isfinite(icrs), axis=1) & (norm > 1e-12)
    icrs[finite] /= norm[finite, None]
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    result[finite, 0] = np.mod(
        np.degrees(np.arctan2(icrs[finite, 1], icrs[finite, 0])),
        360.0,
    )
    result[finite, 1] = np.degrees(
        np.arcsin(np.clip(icrs[finite, 2], -1.0, 1.0))
    )
    return result


def build_control_points(
    model: FrameAstrometricModel,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """采样 Pixel→ICRS 模型并生成 PixInsight 样条控制点。"""

    try:
        grid_size = XISF_EXPORT_MODES[mode][0]
    except KeyError as exc:
        raise ValueError(f"未知的 XISF 控制点模式：{mode}") from exc
    width, height = model.image_width_px, model.image_height_px
    xs = np.linspace(0.0, width - 1.0, grid_size)
    ys = np.linspace(0.0, height - 1.0, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    extra = np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [0.0, height - 1.0],
            [width - 1.0, height - 1.0],
            [width / 2.0, 0.0],
            [width / 2.0, height - 1.0],
            [0.0, height / 2.0],
            [width - 1.0, height / 2.0],
            [width / 2.0, height / 2.0],
        ],
        dtype=np.float64,
    )
    pixels = np.vstack((pixels, extra))
    radec = _extended_equisolid_icrs(model, pixels)
    optical_axis = model.frame_pose.camera_vectors_to_icrs(
        np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64)
    )[0]
    optical_axis /= np.linalg.norm(optical_axis)
    center = np.array(
        [
            np.mod(np.degrees(np.arctan2(optical_axis[1], optical_axis[0])), 360.0),
            np.degrees(np.arcsin(np.clip(optical_axis[2], -1.0, 1.0))),
        ],
        dtype=np.float64,
    )
    world = _zea_world(radec, center)
    valid = np.all(np.isfinite(world), axis=1)
    pixels, world = pixels[valid], world[valid]
    if len(pixels) < 100:
        raise ValueError("天文模型中可用于 XISF 的控制点过少。")
    max_radius = float(np.max(np.linalg.norm(world, axis=1)))
    if max_radius >= 360.0 / np.pi:
        raise ValueError(f"ZEA 控制点超出有效域：{max_radius:.3f}°。")
    return pixels, world, center, max_radius


def _property_item(identifier: str, type_name: str, value: object) -> dict[str, object]:
    return {"id": identifier, "type": type_name, "value": value}


def build_pixinsight_properties(
    model: FrameAstrometricModel,
    pixels: np.ndarray,
    world: np.ndarray,
    center: np.ndarray,
) -> dict[str, dict[str, object]]:
    """构造 PixInsight 可直接识别的 ICRS/ZEA 样条解算属性。"""

    design = np.column_stack((pixels, np.ones(len(pixels))))
    coefficients, *_ = np.linalg.lstsq(design, world, rcond=None)
    matrix = coefficients[:2, :].T.astype(np.float64)
    try:
        reference_image = np.linalg.solve(matrix, -coefficients[2, :]).astype(np.float64)
    except np.linalg.LinAlgError as exc:
        raise ValueError("无法为 XISF 构造稳定的线性参考变换。") from exc

    properties: dict[str, dict[str, object]] = {}

    def add(identifier: str, type_name: str, value: object) -> None:
        properties[identifier] = _property_item(identifier, type_name, value)

    add("Observation:CelestialReferenceSystem", "String", "ICRS")
    add("Observation:Equinox", "Float64", 2000.0)
    add("Observation:Center:RA", "Float64", float(center[0]))
    add("Observation:Center:Dec", "Float64", float(center[1]))
    add("PCL:AstrometricSolution:ProjectionSystem", "String", "ZenithalEqualArea")
    add(
        "PCL:AstrometricSolution:ReferenceCelestialCoordinates",
        "F64Vector",
        center.astype(np.float64),
    )
    add(
        "PCL:AstrometricSolution:ReferenceImageCoordinates",
        "F64Vector",
        reference_image,
    )
    add(
        "PCL:AstrometricSolution:ReferenceNativeCoordinates",
        "F64Vector",
        np.array([0.0, 90.0], dtype=np.float64),
    )
    add(
        "PCL:AstrometricSolution:CelestialPoleNativeCoordinates",
        "F64Vector",
        np.array([180.0 if center[1] < 90.0 else 0.0, 90.0], dtype=np.float64),
    )
    add("PCL:AstrometricSolution:LinearTransformationMatrix", "F64Matrix", matrix)
    add(
        "PCL:AstrometricSolution:CreationTime",
        "TimePoint",
        datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    add("PCL:AstrometricSolution:CreatorApplication", "String", "HoshinoPanoAssistant")
    add("PCL:AstrometricSolution:Catalog", "String", "HoshinoPanoAssistant ICRS model")
    prefix = "PCL:AstrometricSolution:SplineWorldTransformation:"
    add(prefix + "RBFType", "String", "DDMThinPlateSpline")
    add(prefix + "SplineOrder", "Int32", 2)
    add(prefix + "SplineSmoothness", "Float32", 1.0e-6)
    add(prefix + "MaxSplinePoints", "Int32", 4000)
    add(prefix + "UseSimplifiers", "Boolean", "false")
    add(prefix + "SimplifierRejectFraction", "Float32", 0.0)
    add(prefix + "ControlPoints:Image", "F64Vector", pixels.astype(np.float64).ravel())
    add(prefix + "ControlPoints:World", "F64Vector", world.astype(np.float64).ravel())

    hint = model.fit_metadata.get("scene_observer_hint", {})
    if isinstance(hint, dict):
        if "longitude_deg" in hint:
            add("Observation:Location:Longitude", "Float64", float(hint["longitude_deg"]))
        if "latitude_deg" in hint:
            add("Observation:Location:Latitude", "Float64", float(hint["latitude_deg"]))
        if "elevation_m" in hint:
            add("Observation:Location:Elevation", "Float64", float(hint["elevation_m"]))
    return properties


def _xisf_image_array(source: np.ndarray) -> np.ndarray:
    """把项目原生图像数组规范成 python-xisf 支持的 H×W×C 布局。"""

    image = np.asarray(source)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
    if image.ndim != 3:
        raise ValueError(f"XISF 不支持当前图像形状：{image.shape}。")
    channels = int(image.shape[2])
    if channels == 2:
        image = image[:, :, :1]
    elif channels >= 4:
        image = image[:, :, :3]
    elif channels not in (1, 3):
        raise ValueError(f"XISF 不支持当前通道数：{channels}。")
    if image.dtype == np.bool_:
        image = image.astype(np.uint8) * 255
    elif image.dtype == np.float16:
        image = image.astype(np.float32)
    supported_dtypes = {
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.uint32),
        np.dtype(np.float32),
        np.dtype(np.float64),
        np.dtype(np.complex64),
        np.dtype(np.complex128),
    }
    if image.dtype not in supported_dtypes:
        raise ValueError(f"XISF 不支持当前像素类型：{image.dtype}。")
    return np.ascontiguousarray(image)


def verify_xisf_astrometric_solution(path: str | Path) -> None:
    """快速核对成品头部包含 PixInsight 原生样条天文解算。"""

    xisf_path = Path(path)
    with xisf_path.open("rb") as stream:
        prefix = stream.read(16)
        if prefix[:8] != b"XISF0100":
            raise ValueError("导出文件不是有效的 XISF 1.0 文件。")
        header_length = struct.unpack("<I", prefix[8:12])[0]
        header = stream.read(header_length).decode("utf-8")
    missing = [identifier for identifier in PIXINSIGHT_ASTROMETRIC_PROPERTIES if identifier not in header]
    if missing:
        raise ValueError("导出的 XISF 缺少 PixInsight 天文解算属性：" + "、".join(missing))


def export_pixinsight_xisf(
    *,
    image_path: str | Path,
    output_path: str | Path,
    model: FrameAstrometricModel,
    star_pairs: Sequence[Mapping[str, object]],
    mode: str = "fast",
    overwrite: bool = False,
) -> XisfExportResult:
    """写出保留原始位深并包含 PixInsight ICRS 样条解算的 XISF。"""

    source_path = Path(image_path).expanduser().resolve()
    target_path = Path(output_path).expanduser().resolve()
    if target_path.suffix.lower() != ".xisf":
        raise ValueError("XISF 输出文件扩展名必须为 .xisf。")
    if not source_path.is_file():
        raise FileNotFoundError(f"原图不存在：{source_path}")
    if target_path == source_path:
        raise ValueError("XISF 输出路径不能覆盖原图。")
    if target_path.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在：{target_path}")

    validation = validate_star_pairs(model, star_pairs)
    pixels, world, center, max_radius = build_control_points(model, mode)
    properties = build_pixinsight_properties(model, pixels, world, center)
    image = _xisf_image_array(load_native_image_array(source_path))
    actual_size = (int(image.shape[1]), int(image.shape[0]))
    model_size = (model.image_width_px, model.image_height_px)
    if actual_size != model_size:
        raise ValueError(
            f"原图尺寸 {actual_size[0]} x {actual_size[1]} 与天文模型 "
            f"{model_size[0]} x {model_size[1]} 不一致。"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = target_path.with_name(
        f".{target_path.stem}.{uuid4().hex}.partial.xisf"
    )
    try:
        file_size, codec = XISF.write(
            str(temporary_path),
            image,
            creator_app="HoshinoPanoAssistant",
            image_metadata={"XISFProperties": properties},
            codec=None,
        )
        verify_xisf_astrometric_solution(temporary_path)
        os.replace(temporary_path, target_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return XisfExportResult(
        output_path=target_path,
        file_size_bytes=int(file_size),
        codec=codec,
        control_point_count=len(pixels),
        max_zea_radius_deg=max_radius,
        validation=validation,
    )


__all__ = [
    "PIXINSIGHT_ASTROMETRIC_PROPERTIES",
    "XISF_EXPORT_MODES",
    "XisfExportResult",
    "XisfValidationResult",
    "angular_errors_arcsec",
    "build_control_points",
    "build_pixinsight_properties",
    "export_pixinsight_xisf",
    "validate_star_pairs",
    "verify_xisf_astrometric_solution",
]
