"""跨平台文件对话框、类型过滤与快捷键兼容。"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QAbstractItemView, QFileDialog, QFileSystemModel, QShortcut


_SUFFIX_PATTERN = re.compile(r"\*\.([A-Za-z0-9]+)")


def supported_suffixes_from_name_filter(file_filter: str) -> frozenset[str]:
    """从 Qt 名称过滤器提取业务允许的文件后缀。"""

    return frozenset(
        f".{match.group(1).casefold()}"
        for match in _SUFFIX_PATTERN.finditer(file_filter)
    )


def _supported_selected_paths(paths: list[str], file_filter: str) -> list[str]:
    """再次过滤对话框结果，避免平台差异或手工输入绕过文件类型限制。"""

    allowed_suffixes = supported_suffixes_from_name_filter(file_filter)
    if not allowed_suffixes:
        return list(paths)
    return [path for path in paths if Path(path).suffix.casefold() in allowed_suffixes]


def multiple_file_dialog_options(
    *,
    hide_name_filter_details: bool = False,
    platform_name: str | None = None,
) -> QFileDialog.Options:
    """返回多文件选择选项；macOS 使用可控的 Qt 对话框。"""

    options = QFileDialog.Options()
    if hide_name_filter_details:
        options |= QFileDialog.HideNameFilterDetails
    if (platform_name or sys.platform) == "darwin":
        options |= QFileDialog.DontUseNativeDialog
    return options


def _select_all_file_dialog_items(dialog: QFileDialog) -> None:
    """全选对话框当前文件视图，不影响侧边栏和文件类型列表。"""

    for view in dialog.findChildren(QAbstractItemView):
        if not view.isVisible() or view.objectName() not in {"listView", "treeView"}:
            continue
        if view.selectionMode() not in (
            QAbstractItemView.MultiSelection,
            QAbstractItemView.ExtendedSelection,
        ):
            continue
        view.selectAll()


def _hide_unsupported_file_dialog_items(dialog: QFileDialog) -> None:
    """让名称过滤器真正隐藏不匹配文件，而不是只将其禁用。"""

    file_system_model = dialog.findChild(QFileSystemModel)
    if file_system_model is not None:
        file_system_model.setNameFilterDisables(False)


def _add_macos_volume_shortcuts(
    dialog: QFileDialog,
    *,
    volumes_directory: Path = Path("/Volumes"),
) -> None:
    """显式暴露 macOS 挂载点，避免 Qt 把隐藏的 /Volumes 从根目录过滤掉。"""

    if not volumes_directory.is_dir():
        return
    try:
        mounted_paths = sorted(
            (path for path in volumes_directory.iterdir() if path.is_dir()),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        mounted_paths = []

    sidebar_urls = list(dialog.sidebarUrls())
    known_local_paths = {
        str(Path(url.toLocalFile()))
        for url in sidebar_urls
        if url.isLocalFile() and url.toLocalFile()
    }
    for path in (volumes_directory, *mounted_paths):
        path_text = str(path)
        if path_text in known_local_paths:
            continue
        sidebar_urls.append(QUrl.fromLocalFile(path_text))
        known_local_paths.add(path_text)
    dialog.setSidebarUrls(sidebar_urls)


def get_open_file_name(
    parent,
    caption: str,
    directory: str,
    file_filter: str,
    *,
    hide_name_filter_details: bool = False,
    platform_name: str | None = None,
) -> tuple[str, str]:
    """显示单文件对话框；macOS 与多文件入口统一使用 Qt 对话框。"""

    resolved_platform = platform_name or sys.platform
    options = multiple_file_dialog_options(
        hide_name_filter_details=hide_name_filter_details,
        platform_name=resolved_platform,
    )
    if resolved_platform != "darwin":
        return QFileDialog.getOpenFileName(
            parent,
            caption,
            directory,
            file_filter,
            options=options,
        )

    dialog = QFileDialog(parent, caption, directory, file_filter)
    dialog.setOptions(options)
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setFileMode(QFileDialog.ExistingFile)
    _hide_unsupported_file_dialog_items(dialog)
    _add_macos_volume_shortcuts(dialog)
    if dialog.exec_() != QFileDialog.Accepted:
        return "", dialog.selectedNameFilter()
    selected_paths = dialog.selectedFiles()
    return (selected_paths[0] if selected_paths else ""), dialog.selectedNameFilter()


def get_multiple_open_file_names(
    parent,
    caption: str,
    directory: str,
    file_filter: str,
    *,
    hide_name_filter_details: bool = False,
    platform_name: str | None = None,
) -> tuple[list[str], str]:
    """显示多文件对话框；macOS Qt 对话框显式支持 Command+A。"""

    resolved_platform = platform_name or sys.platform
    options = multiple_file_dialog_options(
        hide_name_filter_details=hide_name_filter_details,
        platform_name=resolved_platform,
    )
    if resolved_platform != "darwin":
        selected_paths, selected_filter = QFileDialog.getOpenFileNames(
            parent,
            caption,
            directory,
            file_filter,
            options=options,
        )
        return _supported_selected_paths(selected_paths, file_filter), selected_filter

    dialog = QFileDialog(parent, caption, directory, file_filter)
    dialog.setOptions(options)
    dialog.setAcceptMode(QFileDialog.AcceptOpen)
    dialog.setFileMode(QFileDialog.ExistingFiles)
    _hide_unsupported_file_dialog_items(dialog)
    _add_macos_volume_shortcuts(dialog)
    select_all_shortcut = QShortcut(QKeySequence("Meta+A"), dialog)
    select_all_shortcut.setContext(Qt.WidgetWithChildrenShortcut)
    select_all_shortcut.activated.connect(lambda: _select_all_file_dialog_items(dialog))

    if dialog.exec_() != QFileDialog.Accepted:
        return [], dialog.selectedNameFilter()
    return (
        _supported_selected_paths(dialog.selectedFiles(), file_filter),
        dialog.selectedNameFilter(),
    )


def get_save_file_name(
    parent,
    caption: str,
    directory: str,
    file_filter: str,
    *,
    default_suffix: str = "",
    platform_name: str | None = None,
) -> tuple[str, str]:
    """显示保存对话框；macOS 使用与项目其他入口一致的 Qt 对话框。"""

    resolved_platform = platform_name or sys.platform
    options = multiple_file_dialog_options(platform_name=resolved_platform)
    if resolved_platform != "darwin":
        return QFileDialog.getSaveFileName(
            parent,
            caption,
            directory,
            file_filter,
            options=options,
        )

    dialog = QFileDialog(parent, caption, directory, file_filter)
    dialog.setOptions(options)
    dialog.setAcceptMode(QFileDialog.AcceptSave)
    dialog.setFileMode(QFileDialog.AnyFile)
    _add_macos_volume_shortcuts(dialog)
    if default_suffix:
        dialog.setDefaultSuffix(default_suffix.lstrip("."))
    if dialog.exec_() != QFileDialog.Accepted:
        return "", dialog.selectedNameFilter()
    selected_paths = dialog.selectedFiles()
    return (selected_paths[0] if selected_paths else ""), dialog.selectedNameFilter()


__all__ = [
    "get_open_file_name",
    "get_multiple_open_file_names",
    "get_save_file_name",
    "multiple_file_dialog_options",
    "supported_suffixes_from_name_filter",
]
