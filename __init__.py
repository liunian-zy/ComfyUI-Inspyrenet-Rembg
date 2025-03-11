from .Inspyrenet_Rembg import InspyrenetRembg

NODE_CLASS_MAPPINGS = {
    "InspyrenetRembg" : InspyrenetRembg,
    # "InspyrenetRembgAdvanced" : InspyrenetRembgAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "InspyrenetRembg": "Inspyrenet Rembg",
    # "InspyrenetRembgAdvanced": "Inspyrenet Rembg Advanced"
}
__all__ = ['NODE_CLASS_MAPPINGS', "NODE_DISPLAY_NAME_MAPPINGS"]
