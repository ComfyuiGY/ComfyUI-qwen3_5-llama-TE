# -*- coding: utf-8 -*-
import base64
import gc
import inspect
import io
import os
import re
from dataclasses import dataclass
from functools import wraps

import numpy as np
from PIL import Image

import folder_paths
import comfy.model_management as mm

try:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
except Exception:
    Llama = None
    Qwen3VLChatHandler = None

try:
    from llama_cpp.llama_chat_format import Qwen35ChatHandler
except Exception:
    Qwen35ChatHandler = None


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")

默认图片提示词 = ""
默认图片系统提示词 = "描述这张图,300字左右."
默认文本系统提示词 = "描述这张图,300字左右."


def _确保_llm目录已注册() -> None:
    folder_name = "LLM"
    llm_dir = os.path.join(folder_paths.models_dir, folder_name)

    supported_exts = set(getattr(folder_paths, "supported_pt_extensions", set()))
    llm_exts = supported_exts | {".gguf"}

    try:
        if folder_name not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[folder_name] = ([llm_dir], llm_exts)
            return

        paths, exts = folder_paths.folder_names_and_paths[folder_name]
        if llm_dir not in paths:
            paths.append(llm_dir)

        if isinstance(exts, set):
            exts.update(llm_exts)
        else:
            folder_paths.folder_names_and_paths[folder_name] = (paths, set(exts) | llm_exts)
    except Exception:
        # 不阻断 ComfyUI 启动；后续节点会提示更具体错误
        return


def _列出llm文件() -> list[str]:
    _确保_llm目录已注册()
    try:
        return folder_paths.get_filename_list("LLM")
    except Exception:
        return []


def _图片转base64(image_tensor) -> str:
    """
    编码为 JPEG base64。
    """
    if image_tensor is None:
        return ""

    img = image_tensor[0].cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _缩放图片到最大边(pil: Image.Image, 最大边长: int) -> Image.Image:
    if 最大边长 <= 0:
        return pil
    w, h = pil.size
    long_edge = max(w, h)
    if long_edge <= 最大边长:
        return pil
    scale = 最大边长 / float(long_edge)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return pil.resize((new_w, new_h), resample=Image.BICUBIC)


def _批量图片索引转base64(image_tensor, index: int, 最大边长: int) -> str:
    if image_tensor is None:
        return ""
    if index < 0 or index >= int(image_tensor.shape[0]):
        return ""
    img = image_tensor[index].cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img)
    pil = _缩放图片到最大边(pil, 最大边长)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _调用chat_completion(llm, *, messages, params: dict) -> dict:
    """
    兼容不同 llama-cpp-python 版本的参数名差异（例如 presence_penalty vs present_penalty）。
    """
    kwargs = dict(params or {})
    kwargs["messages"] = messages

    try:
        sig = inspect.signature(llm.create_chat_completion)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except Exception:
        sig = None
        has_var_kw = True

    if sig is not None and not has_var_kw:
        allowed = sig.parameters
        if "presence_penalty" in kwargs and "presence_penalty" not in allowed and "present_penalty" in allowed:
            kwargs["present_penalty"] = kwargs.pop("presence_penalty")
        if "present_penalty" in kwargs and "present_penalty" not in allowed and "presence_penalty" in allowed:
            kwargs["presence_penalty"] = kwargs.pop("present_penalty")
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

    return llm.create_chat_completion(**kwargs)


def _清洗think块文本(text: str) -> str:
    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    cleaned = text
    cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, count=1, flags=re.DOTALL | re.IGNORECASE)

    cleaned = cleaned.replace("<think>", "").replace("</think>", "")
    return cleaned


def _规范化随机种子(seed_value):
    try:
        seed_value = int(seed_value)
    except Exception:
        return None

    if seed_value < 0:
        return None
    return seed_value


def _重置llm推理状态(llm) -> None:
    try:
        ctx = getattr(llm, "_ctx", None)
        if ctx is not None and hasattr(ctx, "memory_clear"):
            ctx.memory_clear(True)
    except Exception:
        pass

    try:
        hybrid_cache_mgr = getattr(llm, "_hybrid_cache_mgr", None)
        if hybrid_cache_mgr is not None and hasattr(hybrid_cache_mgr, "clear"):
            hybrid_cache_mgr.clear()
    except Exception:
        pass

    try:
        batch = getattr(llm, "_batch", None)
        if batch is not None and hasattr(batch, "reset"):
            batch.reset()
    except Exception:
        pass

    try:
        input_ids = getattr(llm, "input_ids", None)
        if input_ids is not None and hasattr(input_ids, "fill"):
            input_ids.fill(0)
    except Exception:
        pass

    try:
        reset = getattr(llm, "reset", None)
        if callable(reset):
            reset()
        elif hasattr(llm, "n_tokens"):
            llm.n_tokens = 0
    except Exception:
        pass


@dataclass
class _QwenModel:
    llm: object
    settings: dict
    chat_handler: object | None = None


class _QwenStorage:
    model: _QwenModel | None = None

    @classmethod
    def unload(cls) -> None:
        try:
            if cls.model and getattr(cls.model.llm, "close", None):
                cls.model.llm.close()
        except Exception:
            pass
        cls.model = None
        gc.collect()
        mm.soft_empty_cache()

    @classmethod
    def load(cls, config: dict) -> _QwenModel:
        if Llama is None:
            raise RuntimeError("未检测到 llama-cpp-python（llama_cpp）。请先安装/更新该依赖。")

        if cls.model and cls.model.settings == config:
            return cls.model

        cls.unload()

        model_path = os.path.join(folder_paths.models_dir, "LLM", config["model"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件：{model_path}")

        mmproj = config.get("mmproj", "无")
        mmproj_path = None
        if mmproj and mmproj != "无":
            mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj)
            if not os.path.exists(mmproj_path):
                raise FileNotFoundError(f"找不到 mmproj 文件：{mmproj_path}")

        family = config["family"]
        think = config["think"]

        chat_handler = None
        if mmproj_path:
            if family == "Qwen3-VL":
                if Qwen3VLChatHandler is None:
                    raise RuntimeError("当前 llama-cpp-python 不支持 Qwen3VLChatHandler，请更新 llama-cpp-python。")
                # Qwen3 的 thinking 参数名在不同版本可能不同，这里做兜底。
                try:
                    chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, force_reasoning=think, verbose=False)
                except Exception:
                    try:
                        chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, use_think_prompt=think, verbose=False)
                    except Exception:
                        chat_handler = Qwen3VLChatHandler(clip_model_path=mmproj_path, verbose=False)
            elif family == "Qwen3.5-VL":
                if Qwen35ChatHandler is None:
                    raise RuntimeError("当前 llama-cpp-python 不支持 Qwen35ChatHandler，请更新 llama-cpp-python。")
                try:
                    chat_handler = Qwen35ChatHandler(
                        clip_model_path=mmproj_path,
                        enable_thinking=think,
                        add_vision_id=True,
                        verbose=False,
                    )
                except TypeError:
                    # 兼容少数版本的参数名差异
                    chat_handler = Qwen35ChatHandler(clip_model_path=mmproj_path, enable_thinking=think, verbose=False)
            else:
                raise ValueError(f"未知模型系列：{family}")

        n_ctx = int(config.get("n_ctx", 8192))
        n_gpu_layers = int(config.get("n_gpu_layers", -1))

        llama_kwargs = {
            "model_path": model_path,
            "chat_handler": chat_handler,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "ctx_checkpoints": 0,
            "verbose": False,
        }

        if mmproj_path:
            vision_batch = max(2048, n_ctx)
            llama_kwargs["n_batch"] = vision_batch
            llama_kwargs["n_ubatch"] = vision_batch

        llm = Llama(**llama_kwargs)

        cls.model = _QwenModel(llm=llm, settings=dict(config), chat_handler=chat_handler)
        return cls.model


def _安装全局卸载挂钩() -> None:
    """
    将 ComfyUI 全局卸载（comfy.model_management.unload_all_models）挂钩到本插件的卸载逻辑上。

    效果：
    - 点击 TEA/ComfyUI 的“释放显存/释放内存”（/free）触发全局卸载时，会同时 close 掉本插件的 llama_cpp 模型。
    """
    try:
        if hasattr(mm, "_qwen_te_unload_hook_installed") and mm._qwen_te_unload_hook_installed:
            return

        original = getattr(mm, "unload_all_models", None)
        if original is None or not callable(original):
            return

        @wraps(original)
        def wrapped_unload_all_models(*args, **kwargs):
            try:
                _QwenStorage.unload()
            except Exception:
                pass
            return original(*args, **kwargs)

        mm.unload_all_models = wrapped_unload_all_models
        mm._qwen_te_unload_hook_installed = True
    except Exception:
        # 不影响 ComfyUI 启动
        return


_安装全局卸载挂钩()


class QwenTE模型加载器:
    @classmethod
    def INPUT_TYPES(s):
        all_files = _列出llm文件()
        model_list = [f for f in all_files if "mmproj" not in f.lower() and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".bin", ".pth", ".pt"]]
        mmproj_list = ["无"] + [f for f in all_files if "mmproj" in f.lower() and os.path.splitext(f)[1].lower() in [".gguf", ".safetensors", ".bin"]]

        if not model_list:
            model_list = ["（请把模型放到 models/LLM）"]

        return {
            "required": {
                "模型系列": (["Qwen3-VL", "Qwen3.5-VL"], {"default": "Qwen3.5-VL"}),
                "主模型": (model_list, {"tooltip": "主模型文件（建议 .gguf）放到 ComfyUI/models/LLM/"}),
                "视觉投影mmproj": (mmproj_list, {"default": "无", "tooltip": "多模态需要 mmproj；纯文本可选“无”。"}),
                "启用思考": ("BOOLEAN", {"default": False, "tooltip": "Qwen3.5: enable_thinking；Qwen3: force_reasoning/use_think_prompt（取决于版本）。"}),
                "上下文长度": ("INT", {"default": 8192, "min": 1024, "max": 327680, "step": 256, "tooltip": "对应 llama.cpp 的 n_ctx。"}),
                "GPU层数": ("INT", {"default": -1, "min": -1, "max": 9999, "step": 1, "tooltip": "对应 llama.cpp 的 n_gpu_layers；-1=尽可能多上GPU；0=纯CPU。"}),
            }
        }

    RETURN_TYPES = ("QWENLLAMA",)
    RETURN_NAMES = ("qwen模型",)
    FUNCTION = "load"
    CATEGORY = "Qwen TE"

    def load(self, 模型系列, 主模型, 视觉投影mmproj, 启用思考, 上下文长度, GPU层数):
        if 主模型.startswith("（请把模型放到"):
            raise RuntimeError("未找到可用模型文件。请把模型放到 ComfyUI/models/LLM/ 后重启。")

        config = {
            "family": 模型系列,
            "model": 主模型,
            "mmproj": 视觉投影mmproj,
            "think": bool(启用思考),
            "n_ctx": int(上下文长度),
            "n_gpu_layers": int(GPU层数),
        }
        model = _QwenStorage.load(config)
        return (model,)


class QwenTE图像推理:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "qwen模型": ("QWENLLAMA",),
                "输入模式": (["图片", "逐帧", "视频", "文本"], {"default": "图片", "tooltip": "图片=只读第1张；逐帧=一张一张推理；视频=抽帧后一次性推理；文本=仅文字输入，无需图片。"}),
                "提示词": ("STRING", {"default": 默认图片提示词, "multiline": True}),
                "系统提示词": ("STRING", {"default": 默认图片系统提示词, "multiline": True}),
                "最多帧数": ("INT", {"default": 24, "min": 2, "max": 1024, "step": 1, "tooltip": "视频模式下从输入图片序列中均匀抽取的帧数。"}),
                "最大边长": ("INT", {"default": 512, "min": 128, "max": 16384, "step": 64, "tooltip": "对输入图片做缩放以提速（取最长边）。"}),
                "最大生成token": ("INT", {"default": 1024, "min": 20, "max": 8192, "step": 1}),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "top_k": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "重复惩罚": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "频率惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "存在惩罚": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1, "control_after_generate": True, "tooltip": "随机种子。可用 ComfyUI 的生成后控制来固定、递增、递减或随机。"}),
                "输出think块": ("BOOLEAN", {"default": True, "tooltip": "开启=保留模型原始 `<think>...</think>` 输出；关闭=仅在最终结果里移除 think 块。"}),
            },
            "optional": {
                "图片": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    FUNCTION = "run"
    CATEGORY = "Qwen TE"

    def run(
        self,
        qwen模型,
        输入模式,
        提示词,
        系统提示词,
        最多帧数,
        最大边长,
        最大生成token,
        温度,
        top_p,
        top_k,
        重复惩罚,
        频率惩罚,
        存在惩罚,
        seed,
        输出think块,
        图片=None,
    ):
        # 卸载后 / 引用失效时：自动重载与同步到当前有效模型
        need_reload = False
        if _QwenStorage.model is None:
            need_reload = True
        elif qwen模型 is not _QwenStorage.model:
            if hasattr(qwen模型, "settings") and getattr(qwen模型, "settings") == _QwenStorage.model.settings:
                qwen模型 = _QwenStorage.model
            else:
                need_reload = True

        if need_reload:
            if not hasattr(qwen模型, "settings"):
                raise RuntimeError("输入的模型对象缺少配置信息，无法自动重载。请先运行“Qwen TE 模型加载器”。")
            _QwenStorage.load(qwen模型.settings)
            qwen模型 = _QwenStorage.model

        if not hasattr(qwen模型, "llm") or qwen模型.llm is None:
            raise RuntimeError("模型对象内部 llm 实例无效，请检查模型文件完整性，或重新加载模型。")

        llm = qwen模型.llm

        messages = []
        system_text = (系统提示词 or "").strip()

        if 输入模式 == "文本":
            if not system_text or system_text == 默认图片系统提示词:
                system_text = 默认文本系统提示词
        elif 输入模式 == "视频" and system_text:
            system_text = "请将输入的图片序列当做视频而不是静态帧序列, " + system_text

        if system_text:
            messages.append({"role": "system", "content": system_text})

        total_images = int(图片.shape[0]) if 图片 is not None else 0
        if 输入模式 in ("图片", "逐帧", "视频") and total_images == 0:
            raise ValueError("未检测到图片输入。")

        if 输入模式 == "图片":
            frame_indices = [0]
        elif 输入模式 == "逐帧":
            frame_indices = list(range(total_images))
        elif 输入模式 == "视频":
            if total_images == 1:
                frame_indices = [0]
            else:
                count = min(max(int(最多帧数), 2), total_images)
                frame_indices = np.linspace(0, total_images - 1, count, dtype=int).tolist()
        elif 输入模式 == "文本":
            frame_indices = []
        else:
            raise ValueError(f"未知输入模式：{输入模式}")

        params = {
            "max_tokens": int(最大生成token),
            "temperature": float(温度),
            "top_p": float(top_p),
            "top_k": int(top_k),
            "repeat_penalty": float(重复惩罚),
            "frequency_penalty": float(频率惩罚),
            "presence_penalty": float(存在惩罚),
            "seed": _规范化随机种子(seed),
            "stream": False,
            "stop": ["</s>"],
        }

        prompt_text = (提示词 or "").strip()
        if 输入模式 == "文本":
            if not prompt_text:
                raise ValueError("文本模式下，提示词不能为空。")

            messages.append({"role": "user", "content": prompt_text})
            _重置llm推理状态(llm)
            out = _调用chat_completion(llm, messages=messages, params=params)
            try:
                text = out["choices"][0]["message"]["content"]
            except Exception:
                text = str(out)
        elif 输入模式 == "逐帧":
            user_content = [{"type": "text", "text": prompt_text}, {"type": "image_url", "image_url": {"url": ""}}]
            messages.append({"role": "user", "content": user_content})

            out_parts = []
            for idx, frame_index in enumerate(frame_indices):
                if mm.processing_interrupted():
                    raise mm.InterruptProcessingException()
                img_b64 = _批量图片索引转base64(图片, frame_index, int(最大边长))
                if not img_b64:
                    continue
                user_content[1]["image_url"]["url"] = f"data:image/jpeg;base64,{img_b64}"
                _重置llm推理状态(llm)
                out = _调用chat_completion(llm, messages=messages, params=params)
                try:
                    part = out["choices"][0]["message"]["content"]
                except Exception:
                    part = str(out)
                if len(frame_indices) > 1:
                    out_parts.append(f"====== 第{idx+1}帧 ======\n{part}".strip())
                else:
                    out_parts.append(str(part).strip())
            text = "\n\n".join([p for p in out_parts if p])
        else:
            user_content = [{"type": "text", "text": prompt_text}]
            for frame_index in frame_indices:
                img_b64 = _批量图片索引转base64(图片, frame_index, int(最大边长))
                if not img_b64:
                    continue
                user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            messages.append({"role": "user", "content": user_content})
            _重置llm推理状态(llm)
            out = _调用chat_completion(llm, messages=messages, params=params)
            try:
                text = out["choices"][0]["message"]["content"]
            except Exception:
                text = str(out)

        if not bool(输出think块):
            text = _清洗think块文本(text)

        if mm.processing_interrupted():
            raise mm.InterruptProcessingException()

        return (text.lstrip().removeprefix(": ").strip(),)


class QwenTE卸载模型:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"任意输入": (any_type,)}}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("任意输出",)
    FUNCTION = "run"
    CATEGORY = "Qwen TE"

    def run(self, 任意输入):
        _QwenStorage.unload()
        return (任意输入,)
