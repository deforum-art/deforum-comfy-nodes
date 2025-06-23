import torch
import cv2
import numpy as np
from PIL import Image, ImageFilter
from .tools import VariantSupport
from .base_node import ImageNode, NODE_POSTFIX

def tensor_to_pil(tensor):
    """Convert ComfyUI image tensor to PIL Image."""
    # ComfyUI images are (batch, height, width, channels) in range [0,1]
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Convert to numpy and scale to 0-255
    np_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(np_image)

def pil_to_tensor(pil_image):
    """Convert PIL Image to ComfyUI image tensor."""
    # Convert to numpy and normalize to 0-1
    np_image = np.array(pil_image).astype(np.float32) / 255.0
    
    # Add batch dimension: (height, width, channels) -> (1, height, width, channels)
    tensor = torch.from_numpy(np_image).unsqueeze(0)
    return tensor

def measure_saturation(img_array):
    """Measure the saturation of an RGB image array."""
    hls = cv2.cvtColor(img_array, cv2.COLOR_RGB2HLS)
    return hls[:, :, 2].mean() / 255.0

@VariantSupport()
class ColorCorrection(ImageNode):
    """
    Apply adaptive color correction to images.
    Corrects saturation and optionally applies white balance.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "saturation_threshold": ("FLOAT", {
                    "default": 0.4, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "max_saturation_factor": ("FLOAT", {
                    "default": 1.3, 
                    "min": 1.0, 
                    "max": 2.0, 
                    "step": 0.1
                }),
                "enable_white_balance": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "white_balance_scale_min": ("FLOAT", {
                    "default": 0.97, 
                    "min": 0.8, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "white_balance_scale_max": ("FLOAT", {
                    "default": 1.03, 
                    "min": 1.0, 
                    "max": 1.2, 
                    "step": 0.01
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_correction"
    
    def apply_correction(self, image, saturation_threshold, max_saturation_factor, 
                        enable_white_balance, white_balance_scale_min=0.97, 
                        white_balance_scale_max=1.03):
        
        # Process each image in the batch
        batch_size = image.shape[0]
        corrected_images = []
        
        for i in range(batch_size):
            # Convert tensor to PIL for processing
            pil_img = tensor_to_pil(image[i:i+1])
            img_array = np.array(pil_img.convert("RGB"))
            
            # Measure saturation
            saturation = measure_saturation(img_array)
            
            if saturation >= saturation_threshold:
                # No correction needed
                corrected_img = pil_img
            else:
                corrected_img_array = img_array.copy()
                
                # Apply white balance correction if enabled
                if enable_white_balance:
                    avg_per_channel = corrected_img_array.reshape(-1, 3).mean(axis=0)
                    avg_intensity = avg_per_channel.mean()
                    scale = avg_intensity / (avg_per_channel + 1e-6)
                    scale = np.clip(scale, white_balance_scale_min, white_balance_scale_max)
                    corrected_img_array = np.clip(corrected_img_array * scale, 0, 255).astype(np.uint8)
                
                # Apply saturation correction
                sat_factor = min(saturation_threshold / (saturation + 1e-6), max_saturation_factor)
                hls = cv2.cvtColor(corrected_img_array, cv2.COLOR_RGB2HLS)
                H, L_chan, S = cv2.split(hls)
                S_boosted = np.clip(S * sat_factor, 0, 255).astype(np.uint8)
                hls_boosted = cv2.merge([H, L_chan, S_boosted])
                corrected_img_array = cv2.cvtColor(hls_boosted, cv2.COLOR_HLS2RGB)
                
                corrected_img = Image.fromarray(corrected_img_array)
            
            # Convert back to tensor
            corrected_tensor = pil_to_tensor(corrected_img)
            corrected_images.append(corrected_tensor)
        
        # Combine batch
        result_tensor = torch.cat(corrected_images, dim=0)
        return (result_tensor,)

@VariantSupport()
class AdaptiveSharpening(ImageNode):
    """
    Apply adaptive sharpening to improve image details and clarity.
    Uses unsharp masking technique with configurable intensity and radius.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1
                }),
                "radius": ("FLOAT", {
                    "default": 1.1, 
                    "min": 0.1, 
                    "max": 3.0, 
                    "step": 0.1
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_sharpening"
    
    def apply_sharpening(self, image, intensity, radius):
        # Process each image in the batch
        batch_size = image.shape[0]
        sharpened_images = []
        
        for i in range(batch_size):
            # Convert tensor to PIL for processing
            pil_img = tensor_to_pil(image[i:i+1])
            
            # Apply sharpening using unsharp masking
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            sharpened = Image.blend(pil_img, blurred, alpha=-intensity)
            
            # Convert back to tensor
            sharpened_tensor = pil_to_tensor(sharpened)
            sharpened_images.append(sharpened_tensor)
        
        # Combine batch
        result_tensor = torch.cat(sharpened_images, dim=0)
        return (result_tensor,)

@VariantSupport()
class ColorCorrectionAndSharpening(ImageNode):
    """
    Combined node that applies both color correction and sharpening in one step.
    Useful for batch processing with consistent settings.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enable_color_correction": ("BOOLEAN", {"default": True}),
                "enable_sharpening": ("BOOLEAN", {"default": True}),
                "saturation_threshold": ("FLOAT", {
                    "default": 0.4, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "max_saturation_factor": ("FLOAT", {
                    "default": 1.3, 
                    "min": 1.0, 
                    "max": 2.0, 
                    "step": 0.1
                }),
                "sharpening_intensity": ("FLOAT", {
                    "default": 0.7, 
                    "min": 0.0, 
                    "max": 2.0, 
                    "step": 0.1
                }),
                "sharpening_radius": ("FLOAT", {
                    "default": 1.1, 
                    "min": 0.1, 
                    "max": 3.0, 
                    "step": 0.1
                }),
            },
            "optional": {
                "enable_white_balance": ("BOOLEAN", {"default": False}),
                "white_balance_scale_min": ("FLOAT", {
                    "default": 0.97, 
                    "min": 0.8, 
                    "max": 1.0, 
                    "step": 0.01
                }),
                "white_balance_scale_max": ("FLOAT", {
                    "default": 1.03, 
                    "min": 1.0, 
                    "max": 1.2, 
                    "step": 0.01
                }),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_combined_processing"
    
    def apply_combined_processing(self, image, enable_color_correction, enable_sharpening,
                                 saturation_threshold, max_saturation_factor,
                                 sharpening_intensity, sharpening_radius,
                                 enable_white_balance=False, white_balance_scale_min=0.97,
                                 white_balance_scale_max=1.03):
        
        processed_image = image
        
        # Apply color correction if enabled
        if enable_color_correction:
            color_corrector = ColorCorrection()
            processed_image = color_corrector.apply_correction(
                processed_image, saturation_threshold, max_saturation_factor,
                enable_white_balance, white_balance_scale_min, white_balance_scale_max
            )[0]
        
        # Apply sharpening if enabled
        if enable_sharpening:
            sharpener = AdaptiveSharpening()
            processed_image = sharpener.apply_sharpening(
                processed_image, sharpening_intensity, sharpening_radius
            )[0]
        
        return (processed_image,)

# -----------------------------------------------------------------------------
#  Registration tables
# -----------------------------------------------------------------------------
IMAGE_NODE_CLASS_MAPPINGS = {
    "ColorCorrection": ColorCorrection,
    "AdaptiveSharpening": AdaptiveSharpening,
    "ColorCorrectionAndSharpening": ColorCorrectionAndSharpening,
}

IMAGE_NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorCorrection": f"Color Correction {NODE_POSTFIX}",
    "AdaptiveSharpening": f"Adaptive Sharpening {NODE_POSTFIX}",
    "ColorCorrectionAndSharpening": f"Color Correction + Sharpening {NODE_POSTFIX}",
} 