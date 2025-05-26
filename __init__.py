from .Inspyrenet_Rembg import InspyrenetRembg, InspyrenetRembgAdvanced
from .Inspyrenet_Rembg_Optimized import InspyrenetRembgOptimized, InspyrenetRembgAdvancedOptimized

NODE_CLASS_MAPPINGS = {
    "InspyrenetRembg" : InspyrenetRembg,
    "InspyrenetRembgAdvanced" : InspyrenetRembgAdvanced,
    "InspyrenetRembgOptimized" : InspyrenetRembgOptimized,
    "InspyrenetRembgAdvancedOptimized" : InspyrenetRembgAdvancedOptimized,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspyrenetRembg": "Inspyrenet Rembg",
    "InspyrenetRembgAdvanced": "Inspyrenet Rembg Advanced",
    "InspyrenetRembgOptimized": "Inspyrenet Rembg (优化版)",
    "InspyrenetRembgAdvancedOptimized": "Inspyrenet Rembg Advanced (优化版)"
}
__all__ = ['NODE_CLASS_MAPPINGS', "NODE_DISPLAY_NAME_MAPPINGS"] 