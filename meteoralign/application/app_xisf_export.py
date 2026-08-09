from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QProgressDialog

from ..frame_astrometry import FrameAstrometricModel
from ..qt_tasks import create_progress_dialog, start_qt_worker_task
from ..xisf_export import (
    XISF_EXPORT_MODES,
    XisfExportResult,
    export_pixinsight_xisf,
    validate_star_pairs,
)
from .file_dialogs import get_save_file_name


XISF_FILE_FILTER = "PixInsight XISF (*.xisf)"
_XISF_MODE_ORDER = ("fast", "lite", "high")


@dataclass(frozen=True)
class ValidatedXisfSourceModel:
    json_path: Path
    model: FrameAstrometricModel
    star_pairs: tuple[dict[str, object], ...]


class XisfExportWorker(QObject):
    """在后台读取原图并写出 XISF，避免大图阻塞 Qt 主线程。"""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        *,
        image_path: Path,
        output_path: Path,
        model: FrameAstrometricModel,
        star_pairs: Sequence[Mapping[str, object]],
        mode: str,
    ) -> None:
        super().__init__()
        self.image_path = Path(image_path)
        self.output_path = Path(output_path)
        self.model = model
        self.star_pairs = tuple(dict(pair) for pair in star_pairs)
        self.mode = mode

    def run(self) -> None:
        try:
            result = export_pixinsight_xisf(
                image_path=self.image_path,
                output_path=self.output_path,
                model=self.model,
                star_pairs=self.star_pairs,
                mode=self.mode,
                overwrite=True,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 - 后台导出错误需要完整反馈到界面。
            self.failed.emit(str(exc))


class XisfExportMixin:
    """星点匹配页的 PixInsight XISF 导出入口。"""

    ui: object
    current_image_preview: object | None
    _source_astrometric_model: object | None
    _xisf_export_thread: object | None
    _xisf_export_worker: object | None
    _xisf_export_progress: QProgressDialog | None
    _xisf_source_model_cache_key: object | None
    _xisf_source_model_cache: ValidatedXisfSourceModel | None

    def _selected_xisf_mode(self) -> str:
        combo = self.ui.comboBoxXisfControlPointMode
        index = int(combo.currentIndex())
        if index < 0 or index >= len(_XISF_MODE_ORDER):
            return "fast"
        return _XISF_MODE_ORDER[index]

    def _default_xisf_path(self, mode: str) -> Path:
        if self.current_image_preview is None:
            raise ValueError("请先导入真实图像。")
        image_path = Path(self.current_image_preview.path).expanduser().resolve()
        return image_path.with_name(f"{image_path.stem}_ICRS_{mode}.xisf")

    def _current_xisf_source_model_path(self) -> Path | None:
        preview = getattr(self, "current_image_preview", None)
        if preview is None:
            return None
        image_path = Path(preview.path).expanduser().resolve()
        path_builder = getattr(self, "_source_model_path_for_image", None)
        if callable(path_builder):
            return Path(path_builder(image_path)).expanduser().resolve()
        return image_path.with_name(f"{image_path.stem}_model.json")

    def _validated_source_model_json_for_xisf(self) -> ValidatedXisfSourceModel | None:
        """返回当前图片旁可用于 XISF 的有效 model.json；结果按文件状态缓存。"""

        preview = getattr(self, "current_image_preview", None)
        model_path = self._current_xisf_source_model_path()
        if preview is None or model_path is None or not model_path.is_file():
            return None
        image_path = Path(preview.path).expanduser().resolve()
        try:
            stat = model_path.stat()
            expected_width = int(preview.original_width)
            expected_height = int(preview.original_height)
        except (OSError, AttributeError, TypeError, ValueError):
            return None
        cache_key = (
            image_path,
            model_path,
            int(stat.st_mtime_ns),
            int(stat.st_size),
            expected_width,
            expected_height,
        )
        if getattr(self, "_xisf_source_model_cache_key", None) == cache_key:
            return getattr(self, "_xisf_source_model_cache", None)

        self._xisf_source_model_cache_key = cache_key
        self._xisf_source_model_cache = None
        try:
            payload = json.loads(model_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            model = FrameAstrometricModel.from_json_payload(payload)
            if (model.image_width_px, model.image_height_px) != (
                expected_width,
                expected_height,
            ):
                return None
            source_image = payload.get("source_image")
            if not isinstance(source_image, dict):
                return None
            recorded_stem = str(source_image.get("file_stem", "")).strip()
            if not recorded_stem:
                recorded_name = str(source_image.get("file_name", "")).strip()
                recorded_stem = Path(recorded_name).stem if recorded_name else ""
            if not recorded_stem or recorded_stem.casefold() != image_path.stem.casefold():
                return None
            pair_payload = payload.get("fit_pairs")
            if not isinstance(pair_payload, list):
                return None
            star_pairs = tuple(dict(pair) for pair in pair_payload if isinstance(pair, dict))
            validate_star_pairs(model, star_pairs)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

        result = ValidatedXisfSourceModel(
            json_path=model_path,
            model=model,
            star_pairs=star_pairs,
        )
        self._xisf_source_model_cache = result
        return result

    def _update_xisf_export_control(self) -> None:
        button = getattr(self.ui, "pushButtonExportXisf", None)
        combo = getattr(self.ui, "comboBoxXisfControlPointMode", None)
        if button is None or combo is None:
            return
        mapping_exported = self._validated_source_model_json_for_xisf() is not None
        idle = getattr(self, "_xisf_export_thread", None) is None
        button.setEnabled(mapping_exported and idle)
        combo.setEnabled(idle)
        if not idle:
            button.setText("正在导出 XISF...")
            button.setToolTip("正在后台写入 XISF，请等待完成")
        elif not mapping_exported:
            button.setText("导出XISF(须先导出映射)")
            button.setToolTip("当前图片旁需要有效的 model.json；没有时请先点击上方“导出映射”")
        else:
            button.setText("导出 XISF")
            button.setToolTip("保留原图位深，导出带 PixInsight ICRS/ZEA 样条天文解算的 XISF")

    def _handle_source_model_written_for_xisf(self, _json_path: Path) -> None:
        """映射写入后清除校验缓存并立即刷新 XISF 按钮。"""

        self._xisf_source_model_cache_key = None
        self._xisf_source_model_cache = None
        self._update_xisf_export_control()

    def export_current_image_xisf(self) -> None:
        if self._xisf_export_thread is not None:
            QMessageBox.information(self, "正在导出 XISF", "当前已有 XISF 正在后台导出，请稍候。")
            return
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入并解析真实图像。")
            return
        validated_model = self._validated_source_model_json_for_xisf()
        if validated_model is None:
            QMessageBox.information(
                self,
                "须先导出映射",
                "当前图片旁没有有效的 model.json。请先点击“导出映射”，或检查已有映射文件。",
            )
            self._update_xisf_export_control()
            return

        mode = self._selected_xisf_mode()
        default_path = self._default_xisf_path(mode)
        selected_path, _selected_filter = get_save_file_name(
            self,
            "导出供 PixInsight 标注的 XISF",
            str(default_path),
            XISF_FILE_FILTER,
            default_suffix="xisf",
        )
        if not selected_path:
            return
        output_path = Path(selected_path).expanduser()
        if output_path.suffix.lower() != ".xisf":
            output_path = output_path.with_suffix(".xisf")

        image_path = Path(self.current_image_preview.path).expanduser().resolve()
        progress = create_progress_dialog(
            self,
            title="正在导出 XISF",
            label_text=(
                "正在读取原始位深图像并写入 PixInsight 天文解算...\n"
                f"{output_path}"
            ),
            minimum=0,
            maximum=0,
        )
        worker = XisfExportWorker(
            image_path=image_path,
            output_path=output_path,
            model=validated_model.model,
            star_pairs=validated_model.star_pairs,
            mode=mode,
        )
        task = start_qt_worker_task(
            parent=self,
            worker=worker,
            finished_signal=worker.finished,
            failed_signal=worker.failed,
            on_finished=self._handle_xisf_export_finished,
            on_failed=self._handle_xisf_export_failed,
            on_cleanup=self._cleanup_xisf_export,
            progress_dialog=progress,
            start_delay_ms=1,
        )
        self._xisf_export_thread = task.thread
        self._xisf_export_worker = task.worker
        self._xisf_export_progress = progress
        self._update_xisf_export_control()
        self.ui.statusbar.showMessage(
            f"正在后台导出 XISF（{XISF_EXPORT_MODES[mode][1]}）: {output_path.name}"
        )

    def _handle_xisf_export_finished(self, result: object) -> None:
        if self._xisf_export_progress is not None:
            self._xisf_export_progress.close()
        export_result = result
        if not isinstance(export_result, XisfExportResult):
            self._handle_xisf_export_failed("XISF 导出返回了无效结果。")
            return
        size_mib = export_result.file_size_bytes / (1024.0 * 1024.0)
        validation = export_result.validation
        self.ui.statusbar.showMessage(f"XISF 已导出: {export_result.output_path}")
        QMessageBox.information(
            self,
            "XISF 已导出",
            (
                f"文件：{export_result.output_path}\n"
                f"大小：{size_mib:.1f} MiB（无压缩）\n"
                f"控制点：{export_result.control_point_count}\n"
                f"ZEA 最大半径：{export_result.max_zea_radius_deg:.3f}°\n"
                f"匹配星反解中位误差：{validation.median_error_arcsec:.2f} arcsec\n\n"
                "可直接在 PixInsight 中打开并使用天文标注。"
            ),
        )

    def _handle_xisf_export_failed(self, error_message: str) -> None:
        if self._xisf_export_progress is not None:
            self._xisf_export_progress.close()
        self.ui.statusbar.showMessage(f"导出 XISF 失败: {error_message}")
        QMessageBox.critical(self, "导出 XISF 失败", error_message)

    def _cleanup_xisf_export(self) -> None:
        self._xisf_export_thread = None
        self._xisf_export_worker = None
        self._xisf_export_progress = None
        self._update_xisf_export_control()


__all__ = [
    "ValidatedXisfSourceModel",
    "XISF_FILE_FILTER",
    "XisfExportMixin",
    "XisfExportWorker",
]
