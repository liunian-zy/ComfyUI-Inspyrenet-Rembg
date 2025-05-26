from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
from tqdm import tqdm
import time


# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class InspyrenetRembg:
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
        print(f"[InspyrenetRembg] 开始处理背景移除，图像数量: {len(image)}")
        
        if (torchscript_jit == "default"):
            remover = Remover()
            print(f"[InspyrenetRembg] 使用默认模式初始化Remover")
        else:
            remover = Remover(jit=True)
            print(f"[InspyrenetRembg] 使用JIT模式初始化Remover")
            
        img_list = []
        for i, img in enumerate(tqdm(image, "Inspyrenet Rembg")):
            img_start = time.time()
            
            # 记录tensor2pil的时间
            t2p_start = time.time()
            pil_img = tensor2pil(img)
            t2p_end = time.time()
            t2p_time = t2p_end - t2p_start
            print(f"[InspyrenetRembg] 图像{i+1}: tensor2pil耗时: {t2p_time:.4f}秒")
            
            # 记录remover.process的时间
            proc_start = time.time()
            mid = remover.process(pil_img, type='rgba')
            proc_end = time.time()
            proc_time = proc_end - proc_start
            print(f"[InspyrenetRembg] 图像{i+1}: remover.process耗时: {proc_time:.4f}秒")
            
            # 记录pil2tensor的时间
            p2t_start = time.time()
            out = pil2tensor(mid)
            p2t_end = time.time()
            p2t_time = p2t_end - p2t_start
            print(f"[InspyrenetRembg] 图像{i+1}: pil2tensor耗时: {p2t_time:.4f}秒")
            
            img_list.append(out)
            img_end = time.time()
            print(f"[InspyrenetRembg] 处理第{i+1}/{len(image)}张图像，总耗时: {img_end - img_start:.4f}秒")
            
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[InspyrenetRembg] 背景移除处理完成，总耗时: {total_time:.4f}秒，平均每张图像: {total_time/len(image):.4f}秒")
        
        return (img_stack, mask)
        
class InspyrenetRembgAdvanced:
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
        print(f"[InspyrenetRembgAdvanced] 开始处理背景移除，图像数量: {len(image)}, 阈值: {threshold}")
        
        if (torchscript_jit == "default"):
            remover = Remover()
            print(f"[InspyrenetRembgAdvanced] 使用默认模式初始化Remover")
        else:
            remover = Remover(jit=True)
            print(f"[InspyrenetRembgAdvanced] 使用JIT模式初始化Remover")
            
        img_list = []
        for i, img in enumerate(tqdm(image, "Inspyrenet Rembg")):
            img_start = time.time()
            
            # 记录tensor2pil的时间
            t2p_start = time.time()
            pil_img = tensor2pil(img)
            t2p_end = time.time()
            t2p_time = t2p_end - t2p_start
            print(f"[InspyrenetRembgAdvanced] 图像{i+1}: tensor2pil耗时: {t2p_time:.4f}秒")
            
            # 记录remover.process的时间
            proc_start = time.time()
            mid = remover.process(pil_img, type='rgba', threshold=threshold)
            proc_end = time.time()
            proc_time = proc_end - proc_start
            print(f"[InspyrenetRembgAdvanced] 图像{i+1}: remover.process耗时: {proc_time:.4f}秒")
            
            # 记录pil2tensor的时间
            p2t_start = time.time()
            out = pil2tensor(mid)
            p2t_end = time.time()
            p2t_time = p2t_end - p2t_start
            print(f"[InspyrenetRembgAdvanced] 图像{i+1}: pil2tensor耗时: {p2t_time:.4f}秒")
            
            img_list.append(out)
            img_end = time.time()
            print(f"[InspyrenetRembgAdvanced] 处理第{i+1}/{len(image)}张图像，总耗时: {img_end - img_start:.4f}秒")
            
        img_stack = torch.cat(img_list, dim=0)
        mask = img_stack[:, :, :, 3]
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[InspyrenetRembgAdvanced] 背景移除处理完成，总耗时: {total_time:.4f}秒，平均每张图像: {total_time/len(image):.4f}秒")
        
        return (img_stack, mask)