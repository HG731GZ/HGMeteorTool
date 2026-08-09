from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox, QProgressDialog

from ..frame_astrometry import FrameAstrometricModel
from ..qt_tasks import create_progress_dialog, start_qt_worker_task
from ..xisf_export import XISF_EXPORT_MODES, XisfExportResult, export_pixinsight_xisf
from .file_dialogs import get_save_file_name


XISF_FILE_FILTER = "PixInsight XISF (*.xisf)"
_XISF_MODE_ORDER = ("fast", "lite", "high")


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
    _xisf_exported_source_model: object | None

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

    def _update_xisf_export_control(self) -> None:
        button = getattr(self.ui, "pushButtonExportXisf", None)
        combo = getattr(self.ui, "comboBoxXisfControlPointMode", None)
        if button is None or combo is None:
            return
        current_model = getattr(self, "_source_astrometric_model", None)
        mapping_exported = (
            getattr(self, "current_image_preview", None) is not None
            and current_model is not None
            and getattr(self, "_xisf_exported_source_model", None) is current_model
        )
        idle = getattr(self, "_xisf_export_thread", None) is None
        button.setEnabled(mapping_exported and idle)
        combo.setEnabled(idle)
        if not idle:
            button.setText("正在导出 XISF...")
            button.setToolTip("正在后台写入 XISF，请等待完成")
        elif not mapping_exported:
            button.setText("导出XISF(须先导出映射)")
            button.setToolTip("请先点击上方“导出映射”，保存当前版本的源图映射")
        else:
            button.setText("导出 XISF")
            button.setToolTip("保留原图位深，导出带 PixInsight ICRS/ZEA 样条天文解算的 XISF")

    def _mark_current_source_model_exported_for_xisf(self, _json_path: Path) -> None:
        """记录当前模型已成功落盘；模型一旦重新求解，身份变化会自动失效。"""

        self._xisf_exported_source_model = getattr(self, "_source_astrometric_model", None)
        self._update_xisf_export_control()

    def _current_source_model_was_exported_for_xisf(self) -> bool:
        current_model = getattr(self, "_source_astrometric_model", None)
        return (
            getattr(self, "current_image_preview", None) is not None
            and current_model is not None
            and getattr(self, "_xisf_exported_source_model", None) is current_model
        )

    def export_current_image_xisf(self) -> None:
        if self._xisf_export_thread is not None:
            QMessageBox.information(self, "正在导出 XISF", "当前已有 XISF 正在后台导出，请稍候。")
            return
        if self.current_image_preview is None:
            QMessageBox.information(self, "尚未导入图像", "请先导入并解析真实图像。")
            return
        if not self._current_source_model_was_exported_for_xisf():
            QMessageBox.information(
                self,
                "须先导出映射",
                "请先点击“导出映射”，保存当前版本的源图映射后再导出 XISF。",
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

        try:
            model_payload = self._build_source_model_payload(output_path)
            model = FrameAstrometricModel.from_json_payload(model_payload)
            pair_payload = model_payload.get("fit_pairs")
            if not isinstance(pair_payload, list):
                raise ValueError("当前天文模型没有可验证的星点匹配记录。")
            star_pairs = [pair for pair in pair_payload if isinstance(pair, dict)]
        except Exception as exc:  # noqa: BLE001 - 模型未就绪时应直接向用户说明原因。
            self.ui.statusbar.showMessage(f"准备 XISF 导出失败: {exc}")
            QMessageBox.critical(self, "无法导出 XISF", str(exc))
            return

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
            model=model,
            star_pairs=star_pairs,
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


__all__ = ["XISF_FILE_FILTER", "XisfExportMixin", "XisfExportWorker"]
