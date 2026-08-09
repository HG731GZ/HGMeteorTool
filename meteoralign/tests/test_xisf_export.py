from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile

from meteoralign.camera_calibration import CameraCalibrationProfile
from meteoralign.frame_astrometry import FrameAstrometricModel, FramePose
from meteoralign.xisf_export import (
    PIXINSIGHT_ASTROMETRIC_PROPERTIES,
    build_control_points,
    export_pixinsight_xisf,
    verify_xisf_astrometric_solution,
)
from meteoralign.application.app_xisf_export import XisfExportMixin


class _FakeControl:
    def __init__(self) -> None:
        self.enabled = True
        self.text = ""
        self.tooltip = ""

    def setEnabled(self, enabled: bool) -> None:  # noqa: N802 - 模拟 Qt 接口。
        self.enabled = bool(enabled)

    def setText(self, text: str) -> None:  # noqa: N802 - 模拟 Qt 接口。
        self.text = text

    def setToolTip(self, text: str) -> None:  # noqa: N802 - 模拟 Qt 接口。
        self.tooltip = text


class _XisfControlHarness(XisfExportMixin):
    def __init__(self, image_path: Path, width: int = 16, height: int = 12) -> None:
        self.ui = SimpleNamespace(
            pushButtonExportXisf=_FakeControl(),
            comboBoxXisfControlPointMode=_FakeControl(),
        )
        self.current_image_preview = SimpleNamespace(
            path=image_path,
            original_width=width,
            original_height=height,
        )
        self._xisf_source_model_cache_key = None
        self._xisf_source_model_cache = None
        self._xisf_export_thread = None


def _rectilinear_model(width: int, height: int) -> FrameAstrometricModel:
    profile = CameraCalibrationProfile(
        image_width_px=width,
        image_height_px=height,
        base_projection_type="rectilinear",
        principal_point_x_px=(width - 1.0) / 2.0,
        principal_point_y_px=(height - 1.0) / 2.0,
        scale_x_px=float(width),
        scale_y_px=float(width),
        global_distortion_type="none",
    )
    return FrameAstrometricModel(
        image_width_px=width,
        image_height_px=height,
        frame_pose=FramePose(np.eye(3, dtype=np.float64)),
        camera_calibration_profile=profile,
        fit_metadata={
            "scene_observer_hint": {
                "longitude_deg": 116.0,
                "latitude_deg": 40.0,
                "elevation_m": 100.0,
            }
        },
    )


def _star_pairs(model: FrameAstrometricModel) -> list[dict[str, float]]:
    pixels = np.asarray(
        [[2.0, 2.0], [13.0, 2.0], [2.0, 9.0], [13.0, 9.0], [7.5, 5.5]],
        dtype=np.float64,
    )
    radec = model.pixel_to_sky_points(pixels)
    return [
        {
            "image_x_px": float(pixel[0]),
            "image_y_px": float(pixel[1]),
            "ra_deg": float(sky[0]),
            "dec_deg": float(sky[1]),
        }
        for pixel, sky in zip(pixels, radec)
    ]


def _write_model_json(path: Path, image_path: Path, model: FrameAstrometricModel) -> None:
    payload = model.to_json_payload(
        source_image={
            "file_name": image_path.name,
            "file_stem": image_path.stem,
            "original_width_px": model.image_width_px,
            "original_height_px": model.image_height_px,
        },
        fit_pairs=_star_pairs(model),
    )
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_export_pixinsight_xisf_preserves_uint16_and_writes_solution(tmp_path: Path) -> None:
    width, height = 16, 12
    model = _rectilinear_model(width, height)
    source_path = tmp_path / "source.tif"
    image = np.arange(width * height * 3, dtype=np.uint16).reshape(height, width, 3)
    tifffile.imwrite(source_path, image, photometric="rgb")
    output_path = tmp_path / "source_ICRS_fast.xisf"

    result = export_pixinsight_xisf(
        image_path=source_path,
        output_path=output_path,
        model=model,
        star_pairs=_star_pairs(model),
        mode="fast",
    )

    assert result.output_path == output_path
    assert result.control_point_count == 634
    assert result.validation.validated_count == 5
    assert result.validation.median_error_arcsec < 0.01
    assert output_path.stat().st_size == result.file_size_bytes
    verify_xisf_astrometric_solution(output_path)
    header = output_path.read_bytes()[:200_000].decode("utf-8", errors="ignore")
    assert 'sampleFormat="UInt16"' in header
    assert all(identifier in header for identifier in PIXINSIGHT_ASTROMETRIC_PROPERTIES)


def test_existing_equisolid_model_extends_fast_grid_to_all_rectangle_corners() -> None:
    project_root = Path(__file__).resolve().parents[2]
    model = FrameAstrometricModel.from_json_file(
        str(project_root / "testimages" / "A7R3_1214_DSC05361_model.json")
    )

    pixels, world, _center, max_radius = build_control_points(model, "fast")

    assert len(pixels) == 634
    assert np.all(np.isfinite(world))
    assert max_radius < 360.0 / np.pi


def test_xisf_button_accepts_existing_valid_model_json(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.tif"
    model_path = tmp_path / "frame_model.json"
    model = _rectilinear_model(16, 12)
    _write_model_json(model_path, image_path, model)
    harness = _XisfControlHarness(image_path)

    harness._update_xisf_export_control()
    assert harness.ui.pushButtonExportXisf.enabled
    assert harness.ui.pushButtonExportXisf.text == "导出 XISF"

    model_path.write_text("{损坏的 JSON", encoding="utf-8")
    harness._update_xisf_export_control()
    assert not harness.ui.pushButtonExportXisf.enabled
    assert harness.ui.pushButtonExportXisf.text == "导出XISF(须先导出映射)"


def test_model_json_for_another_image_does_not_unlock_xisf(tmp_path: Path) -> None:
    image_path = tmp_path / "frame.tif"
    model = _rectilinear_model(16, 12)
    _write_model_json(tmp_path / "frame_model.json", tmp_path / "other.tif", model)
    harness = _XisfControlHarness(image_path)

    harness._update_xisf_export_control()

    assert not harness.ui.pushButtonExportXisf.enabled


def test_programmatic_xisf_export_cannot_bypass_mapping_gate(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    harness = _XisfControlHarness(tmp_path / "frame.tif")
    dialogs: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "meteoralign.application.app_xisf_export.QMessageBox.information",
        lambda _parent, title, message: dialogs.append((title, message)),
    )

    harness.export_current_image_xisf()

    assert dialogs == [
        (
            "须先导出映射",
            "当前图片旁没有有效的 model.json。请先点击“导出映射”，或检查已有映射文件。",
        )
    ]
