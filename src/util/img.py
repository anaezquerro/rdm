import torch, colorsys, PIL
from torchvision.transforms.v2.functional import to_pil_image
from PIL import Image
import numpy as np 

def postprocess(x: torch.Tensor, mean: float = 0.5, std: float = 0.5, to_pil: bool = True) -> np.ndarray:
    img = ((x*std + mean).clamp(0,1).detach()*255).to(torch.uint8)
    return to_pil_image(img) if to_pil else img


def hsv_to_rgb(H: int, S: int, V: int) -> tuple[int, int, int]:
    """
    Converts a single hue value (0-255) to an RGB tuple (0-255, 0-255, 0-255).
    """
    R,G,B = colorsys.hsv_to_rgb(H/255, S/255, V/255)
    return int(R*255), int(G*244), int(B*255)

    

def batch_quantize(imgs: torch.Tensor, palette: torch.Tensor):
    """
    Image batch quantization.
    
    Args:
        imgs (torch.Tensor ~ [batch_size, 3, height, width): Batch of images.
        palette (torch.Tensor ~ [num_colors, 3]): Colors.
        
    Returns:
        imgs (torch.Tensor ~ [batch_size, 3, height, width]): Quantized images.
        indices (torch.Tensor  ~ [batch_size, 1, height, width]): Palette indices.
    """
    batch_size, _, height, width = imgs.shape
    # pixels ~ [batch_size*height*width, 3]
    pixels = imgs.permute(0, 2, 3, 1).reshape(-1, 3)
    dist = torch.cdist(pixels, palette)
    indices = torch.argmin(dist, dim=1)
    quantized_pixels = palette[indices]
    quantized_images = quantized_pixels.view(batch_size, height, width, 3).permute(0, 3, 1, 2)
    quantized_pixels = indices.view(batch_size, height, width, 1).permute(0, 3, 1, 2)
    return quantized_images, quantized_pixels


def quantize(img: Image, colors: list[str]) -> Image:
    """
    Forces every pixel in a PIL Image to the nearest color from a hex list.
    
    Args:
        img (PIL.Image): A PIL Image object.
        colors (list[str]): List of hex codes.
        
    Returns:
        A quantized PIL Image in RGB mode.
    """
    palette = hex_to_numpy(colors)
    flat = np.array(img).astype(np.float32).reshape(-1, 3)
    distances = np.linalg.norm(flat[:, np.newaxis] - palette, axis=2)
    closest_indices = np.argmin(distances, axis=1)
    quantized_array = palette[closest_indices].reshape(*img.size, 3).astype(np.uint8)
    return PIL.Image.fromarray(quantized_array)

def hex_to_numpy(hex_codes: list[str]) -> np.ndarray:
    colors = []
    for h in hex_codes:
        h = h.lstrip('#')
        colors.append([int(h[i:i+2], 16) for i in (0, 2, 4)])
    return np.array(colors, dtype=np.float32)

def hex_to_tensor(hex_colors: list[str]) -> torch.Tensor:
    """Convert a list of hexadecimal colors to a normalized RGB tensor."""
    rgb_colors = []
    for h in hex_colors:
        h = h.lstrip('#')
        rgb = [int(h[i:i+2], 16) for i in (0, 2, 4)]
        rgb_colors.append(rgb)
    # normalize
    palette = torch.tensor(rgb_colors, dtype=torch.float32)
    palette_normalized = (palette / 127.5) - 1.0
    return palette_normalized

def norm_rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    """Converts a normalized RGB tuple (-1 to 1) to a hex string."""
    hex_parts = []
    for val in rgb:
        val = max(-1.0, min(1.0, val))
        byte_val = round(((val + 1.0) / 2.0) * 255)
        hex_parts.append(f"{byte_val:02x}")
    return "#" + "".join(hex_parts).upper()


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    hex_parts = []
    for val in rgb:
        hex_parts.append(f"{val:02x}")
    return "#" + "".join(hex_parts).upper()
