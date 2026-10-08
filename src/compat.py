"""兼容性工具: 修复旧版 TensorFlow(~2.0/2.1) + 新版 h5py(>=3.0) 组合下
model.load_weights() 报错 `AttributeError: 'str' object has no attribute
'decode'` 的问题。

背景: TF 2.0 时代的 load_weights 内部代码假设 h5py 的属性(attrs)读出来
是 bytes(旧版 h5py<3.0 的默认行为),而 h5py>=3.0 默认直接返回 str /
字符串数组,导致内部 `.decode('utf8')` 调用失败。这不是模型或权重文件
本身的问题,只是训练/推理环境的包版本差异导致的。

什么时候需要用它: 如果你用来 **训练** 模型的环境 TensorFlow 版本
>= 2.4(能正常识别 `tf.keras.layers.MultiHeadAttention`,本项目融合层
用到了这个层),之后也在同一个环境或另一个同样 TF>=2.4 的环境里加载
权重做推理,通常不需要这个补丁,新版 TF 自己的 load_weights 代码已经
适配了新版 h5py。只有当你被迫用一个更旧的 TF 环境(比如没有GPU/CUDA
驱动支持更高版本)去加载权重时,才可能碰到这个报错,这时候在加载权重
之前调用一下 patch_h5py_for_legacy_tf() 即可。

用法:
    from src.compat import patch_h5py_for_legacy_tf
    patch_h5py_for_legacy_tf()
    model.load_weights(...)
"""
import numpy as np


def patch_h5py_for_legacy_tf():
    """给 h5py.AttributeManager 打补丁,让属性读取行为退回旧版(bytes)。

    只在当前 Python 进程内生效,不修改环境里实际安装的 h5py 版本,
    也不影响其它依赖 h5py 的代码;重复调用是安全的(内部做了幂等处理)。
    """
    import h5py

    if getattr(h5py.AttributeManager, '_dcdsm_patched', False):
        return  # 已经打过补丁,避免重复包裹

    orig_getitem = h5py.AttributeManager.__getitem__

    def _encode_strs(value):
        if isinstance(value, str):
            return value.encode('utf8')
        if isinstance(value, bytes):
            return value
        if isinstance(value, np.ndarray):
            if value.dtype.kind in ('U', 'O', 'S'):
                return np.array([_encode_strs(v) for v in value.tolist()], dtype=object)
            return value
        if isinstance(value, (list, tuple)):
            return type(value)(_encode_strs(v) for v in value)
        return value

    def patched_getitem(self, name):
        return _encode_strs(orig_getitem(self, name))

    h5py.AttributeManager.__getitem__ = patched_getitem
    h5py.AttributeManager._dcdsm_patched = True
    print("[兼容性] 已对 h5py.AttributeManager 打补丁,修复旧版TF + 新版h5py 的 decode 报错")
