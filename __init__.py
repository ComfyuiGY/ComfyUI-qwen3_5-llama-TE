# -*- coding: utf-8 -*-
"""
ComfyUI  Qwen3/Qwen3.5 llama TE 插件
"""

from .nodes import (
    QwenTE模型加载器,
    QwenTE图像推理,
    QwenTE卸载模型,
)

NODE_CLASS_MAPPINGS = {
    "QwenTE_ModelLoader": QwenTE模型加载器,
    "QwenTE_ImageInfer": QwenTE图像推理,
    "QwenTE_Unload": QwenTE卸载模型,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenTE_ModelLoader": "Qwen TE 模型加载器",
    "QwenTE_ImageInfer": "Qwen TE 图像推理",
    "QwenTE_Unload": "Qwen TE 卸载模型",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
