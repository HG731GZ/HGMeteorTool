# HoshinoPanoAssistant

面向星野与流星雨摄影的桌面处理工具，用于流星框选、星空模拟、星点匹配、图像序列解析、自由投影构图和全景批处理。项目的 Python 包名为 `meteoralign`，图形界面基于 PyQt5，主要计算可在本地离线完成。

> 当前项目面向 Windows 与 macOS 双端，支持在两种系统上从源码运行；界面语言为简体中文。

## 主要功能

- **流星框选**：批量导入普通图片或相机 RAW，手动框选流星并保存关联 JSON；可使用蒙版，也可连接 MetDet worker 自动检测。
- **星空模拟**：根据拍摄时间、经纬度、海拔、传感器尺寸、焦距和镜头类型生成星空视野，显示恒星、星座、银河、太阳系天体及可选的流星雨辐射点。
- **星点匹配**：将模拟参考星图与真实图像配对，通过 PSF 拟合、自动扩展匹配和局部残差建立源图映射模型。
- **图像序列解析**：利用首帧解算结果和 EXIF 时间批量分析固定机位序列，并支持逐帧姿态修正与结果续算。
- **全景构图**：把多张已解算图像投影到统一天球画布，预览构图、计算输出尺寸、保存取景，并导出重投影 TIFF。
- **全景图批处理**：按统一取景或底图批量重投影图像，可仅导出流星区域，并提供低频周边梯度优化。
- **多种投影模型**：支持 TAN、ARC、ZEA、STG、MER、CAR 以及普适锚点插值。
- **PixInsight 衔接**：可导出带天文坐标控制点的 XISF，提供多档控制点密度。

## 典型工作流

```text
星空模拟并导出参考图
        ↓
导入真实图像并匹配参考星
        ↓
导出每张图像的源图模型 JSON
        ↓
在“全景构图”中设计并导出取景 JSON
        ↓
在“全景图批处理”中统一重投影
```

如果只需要整理流星素材，可直接在“框选流星”页完成导入、检测、框选和文件移动，不必执行后续星点匹配流程。

## 环境要求

- Python 3.10
- Conda 或兼容的环境管理器
- Windows 或 macOS 桌面环境
- 足够的内存与磁盘空间；高分辨率 TIFF 重投影的资源占用会随输出尺寸增加

项目已固定主要依赖版本，推荐使用仓库中的 `environment.yml`，以避免 Qt、NumPy、Astropy、OpenCV 和图像编解码库之间的兼容问题。

## 安装

在项目根目录创建并激活环境：

```bash
conda env create -f environment.yml
conda activate hgastro
```

如果 `hgastro` 环境已经存在：

```bash
conda env update -n hgastro -f environment.yml --prune
conda activate hgastro
```

检查解释器、依赖版本和 Qt 是否可用：

```bash
python scripts/verify_hgastro_env.py
```

也可以在自行管理的 Python 3.10 环境中安装依赖：

```bash
python -m pip install -r requirements.txt
```

## 准备离线星表

程序启动时需要 `catalog/` 中的恒星、星名、银河轮廓和太阳系历表数据。仓库若未包含完整数据，可执行：

```bash
python scripts/download_catalogs.py
```

下载内容包括 Yale Bright Star Catalog、Hipparcos 主星表、IAU 恒星名称表、d3-celestial 银河数据和 JPL DE440s 历表。下载完成后，日常模拟与解算无需联网。

如需重新下载或修复不完整文件：

```bash
python scripts/download_catalogs.py --force
```

## 启动

```bash
python main.py
```

也可以通过包入口启动：

```bash
python -m meteoralign
```

首次启动会创建或补全 `preference.json`。源码运行时该文件位于项目根目录；打包程序使用当前用户的系统配置目录。大多数选项可在程序的“软件参数”窗口中调整并保存。

## 图像与输出格式

常规星点匹配、序列解析和重投影流程支持：

- TIFF：`.tif`、`.tiff`
- JPEG：`.jpg`、`.jpeg`
- PNG：`.png`

“框选流星”页还可通过 LibRaw 读取常见相机 RAW 格式，例如 DNG、CR2、CR3、NEF、ARW、RAF、ORF、RW2 等。高位深重投影结果以 TIFF 输出；星点匹配页还可导出 XISF 和关联的模型 JSON。

建议不要移动或重命名已参与解算的原图、模型 JSON、流星框选 JSON 和取景 JSON。模型会优先使用记录的相对路径查找原图，但保持文件结构不变最可靠。

## 可选：自动检测流星

手动框选不需要额外组件。自动检测需要兼容 `metdet.jsonl` v1 协议的 MetDet worker；可在“自动检测选项”中选择 `metdet_worker`、`metdet_worker.exe`、`metdet_worker.py` 或其所在目录，并配置模型文件和推理参数。

MetDet worker 不包含在 Python 依赖安装步骤中，未配置时不影响其他功能。

## 开发与测试

运行全部测试：

```bash
python -m pytest meteoralign/tests
```

运行单个测试文件：

```bash
python -m pytest meteoralign/tests/test_projection_alignment.py
```

项目包含较多 Qt 界面测试；无显示服务器的环境可先设置 `QT_QPA_PLATFORM=offscreen`。

## 打包发布

项目支持 Windows 与 macOS 双端运行。当前仓库提供了 Windows 自动打包脚本，可使用 PyInstaller 构建 onedir 发布目录：

```bash
python scripts/build_windows.py
```

构建前需确认：

- `catalog/`、`qrcode/`、`hooks/` 和 `icon256.png` 完整存在；
- 已准备 Windows 版 `metdet_worker`；
- worker 默认位于 `MetDetPy/metdet_worker`，也可通过环境变量 `METDET_WORKER_DIR` 指定。

产物输出到 `dist/HoshinoPanoAssistant-Win-<架构>/HoshinoPanoAssistant/`。

## 项目结构

```text
HGMeteorTool/
├── main.py                 # 桌面程序入口
├── meteoralign/            # 核心算法、界面与测试
│   ├── alignment/          # 天文投影与星点配准
│   ├── application/        # PyQt 应用逻辑
│   ├── mosaic/             # 全景预览与重投影
│   ├── photometric/        # 低频光度/梯度校正
│   ├── projection/         # 相机与投影模型
│   └── tests/              # pytest 测试
├── catalog/                # 运行所需的离线天文数据
├── scripts/                # 下载、环境检查、基准与构建脚本
├── hooks/                  # PyInstaller hooks
├── environment.yml         # 推荐 Conda 环境
└── requirements.txt        # pip 依赖
```

## 注意事项

- 经纬度约定为北纬、东经为正；请同时确认 UTC 偏移和拍摄时间。
- EXIF 时间缺少时区信息时，应在界面中手动核对，时间误差会直接影响星空位置。
- 高精度投影、Lanczos3 插值、较小投影网格和大尺寸输出会显著增加处理时间。
- 重要素材建议先备份；“移动流星文件”会移动原图及其同目录关联 JSON。
- 仓库目前未提供独立许可证文件；在复制、分发或用于商业项目之前，请先向项目作者确认授权范围。
