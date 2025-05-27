from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm
import time

import os
import folder_paths

os.environ['TRANSPARENT_BACKGROUND_FILE_PATH'] = os.path.join(folder_paths.models_dir, "transparent_background")
ckpt_path = os.path.join(folder_paths.models_dir, "transparent_background", ".transparent-background", "ckpt_base.pth")
ckpt_path = None if not os.path.exists(ckpt_path) else ckpt_path
# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class InspyrenetRembgOptimized:
    # 类变量存储Remover实例
    _remover_default = None
    _remover_jit = None
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit):
        start_time = time.time()
        print(f"[InspyrenetRembgOptimized] 开始处理背景移除，图像数量: {len(image)}")
        
        # 懒加载Remover实例，只在首次需要时创建
        if (torchscript_jit == "default"):
            if InspyrenetRembgOptimized._remover_default is None:
                remover_init_start = time.time()
                print(f"[InspyrenetRembgOptimized] 首次初始化默认Remover...")
                InspyrenetRembgOptimized._remover_default = Remover(ckpt=ckpt_path)
                remover_init_end = time.time()
                print(f"[InspyrenetRembgOptimized] Remover初始化耗时: {remover_init_end - remover_init_start:.4f}秒")
            else:
                print(f"[InspyrenetRembgOptimized] 使用已缓存的默认Remover实例")
            remover = InspyrenetRembgOptimized._remover_default
        else:
            if InspyrenetRembgOptimized._remover_jit is None:
                remover_init_start = time.time()
                print(f"[InspyrenetRembgOptimized] 首次初始化JIT Remover...")
                InspyrenetRembgOptimized._remover_jit = Remover(jit=True,ckpt=ckpt_path)
                remover_init_end = time.time()
                print(f"[InspyrenetRembgOptimized] Remover JIT初始化耗时: {remover_init_end - remover_init_start:.4f}秒")
            else:
                print(f"[InspyrenetRembgOptimized] 使用已缓存的JIT Remover实例")
            remover = InspyrenetRembgOptimized._remover_jit
            
        img_list = []
        for i, img in enumerate(tqdm(image, "Inspyrenet Rembg")):
            img_start = time.time()
            
            # 记录tensor2pil的时间
            t2p_start = time.time()
            pil_img = tensor2pil(img)
            t2p_end = time.time()
            t2p_time = t2p_end - t2p_start
            print(f"[InspyrenetRembgOptimized] 图像{i+1}: tensor2pil耗时: {t2p_time:.4f}秒")
            
            # 记录remover.process的时间
            proc_start = time.time()
            mid = remover.process(pil_img, type='rgba')
            proc_end = time.time()
            proc_time = proc_end - proc_start
            print(f"[InspyrenetRembgOptimized] 图像{i+1}: remover.process耗时: {proc_time:.4f}秒")
            
            # 记录pil2tensor的时间
            p2t_start = time.time()
            out = pil2tensor(mid)
            p2t_end = time.time()
            p2t_time = p2t_end - p2t_start
            print(f"[InspyrenetRembgOptimized] 图像{i+1}: pil2tensor耗时: {p2t_time:.4f}秒")
            
            img_list.append(out)
            img_end = time.time()
            print(f"[InspyrenetRembgOptimized] 处理第{i+1}/{len(image)}张图像，总耗时: {img_end - img_start:.4f}秒")
            
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[InspyrenetRembgOptimized] 背景移除处理完成，总耗时: {total_time:.4f}秒，平均每张图像: {total_time/len(image):.4f}秒")
        
        return (img_stack, mask)
        
class InspyrenetRembgAdvancedOptimized:
    # 类变量存储Remover实例
    _remover_default = None
    _remover_jit = None
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "torchscript_jit": (["default", "on"],)
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image, torchscript_jit, threshold):
        start_time = time.time()
        print(f"[InspyrenetRembgAdvancedOptimized] 开始处理背景移除，图像数量: {len(image)}, 阈值: {threshold}")
        
        # 懒加载Remover实例，只在首次需要时创建
        if (torchscript_jit == "default"):
            if InspyrenetRembgAdvancedOptimized._remover_default is None:
                remover_init_start = time.time()
                print(f"[InspyrenetRembgAdvancedOptimized] 首次初始化默认Remover...")
                InspyrenetRembgAdvancedOptimized._remover_default = Remover(ckpt=ckpt_path)
                remover_init_end = time.time()
                print(f"[InspyrenetRembgAdvancedOptimized] Remover初始化耗时: {remover_init_end - remover_init_start:.4f}秒")
            else:
                print(f"[InspyrenetRembgAdvancedOptimized] 使用已缓存的默认Remover实例")
            remover = InspyrenetRembgAdvancedOptimized._remover_default
        else:
            if InspyrenetRembgAdvancedOptimized._remover_jit is None:
                remover_init_start = time.time()
                print(f"[InspyrenetRembgAdvancedOptimized] 首次初始化JIT Remover...")
                InspyrenetRembgAdvancedOptimized._remover_jit = Remover(jit=True,ckpt=ckpt_path)
                remover_init_end = time.time()
                print(f"[InspyrenetRembgAdvancedOptimized] Remover JIT初始化耗时: {remover_init_end - remover_init_start:.4f}秒")
            else:
                print(f"[InspyrenetRembgAdvancedOptimized] 使用已缓存的JIT Remover实例")
            remover = InspyrenetRembgAdvancedOptimized._remover_jit
            
        img_list = []
        for i, img in enumerate(tqdm(image, "Inspyrenet Rembg")):
            img_start = time.time()
            
            # 记录tensor2pil的时间
            t2p_start = time.time()
            pil_img = tensor2pil(img)
            t2p_end = time.time()
            t2p_time = t2p_end - t2p_start
            print(f"[InspyrenetRembgAdvancedOptimized] 图像{i+1}: tensor2pil耗时: {t2p_time:.4f}秒")
            
            # 记录remover.process的时间
            proc_start = time.time()
            mid = remover.process(pil_img, type='rgba', threshold=threshold)
            proc_end = time.time()
            proc_time = proc_end - proc_start
            print(f"[InspyrenetRembgAdvancedOptimized] 图像{i+1}: remover.process耗时: {proc_time:.4f}秒")
            
            # 记录pil2tensor的时间
            p2t_start = time.time()
            out = pil2tensor(mid)
            p2t_end = time.time()
            p2t_time = p2t_end - p2t_start
            print(f"[InspyrenetRembgAdvancedOptimized] 图像{i+1}: pil2tensor耗时: {p2t_time:.4f}秒")
            
            img_list.append(out)
            img_end = time.time()
            print(f"[InspyrenetRembgAdvancedOptimized] 处理第{i+1}/{len(image)}张图像，总耗时: {img_end - img_start:.4f}秒")
            
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[InspyrenetRembgAdvancedOptimized] 背景移除处理完成，总耗时: {total_time:.4f}秒，平均每张图像: {total_time/len(image):.4f}秒")
        
        return (img_stack, mask) 