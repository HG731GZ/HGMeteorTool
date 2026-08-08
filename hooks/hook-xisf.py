from PyInstaller.utils.hooks import copy_metadata


# xisf.py 在导入时通过 importlib.metadata.version("xisf") 读取自身版本。
datas = copy_metadata("xisf")
