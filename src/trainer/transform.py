from torchvision.transforms.v2 import Normalize, Compose, Resize, RandomPerspective
import torch, PIL, cv2, random
import numpy as np
from torchvision.transforms.v2.functional import pil_to_tensor, pad, to_pil_image, resize, InterpolationMode, perspective, crop, resize
from transformers import AutoImageProcessor


class Transform:

    def __init__(
        self,
        name: str, 
        source: list[str],
        img_size: tuple[int, int, int],
        pretrained: str = None,
        interp: str = 'bilinear'
    ):
        self.name = name 
        self.source = source
        self.pretrained = pretrained
        self.interp = interp
        self.img_size = self.num_channels, *self.resolution = img_size

        if pretrained:
            self.proc = AutoImageProcessor.from_pretrained(pretrained, use_fast=True)
            self.transform = self.apply_proc
        else:
            self.transform = Compose([
                self.to_tensor, 
                Normalize([0.5], [0.5]),
                Resize(self.resolution, InterpolationMode.BILINEAR if interp == 'bilinear' else InterpolationMode.NEAREST)
            ])
            
    def apply_proc(self, x: torch.Tensor):
        if x.shape[0] == 1:
            x = x.repeat(3, 1, 1)
        return self.proc(x, return_tensors='pt').pixel_values.squeeze(0)
    
    def __repr__(self) -> str:
        return f'Transform({self.name}, resolution={self.resolution}, interp={self.interp})'

    def to_tensor(self, x: torch.Tensor | PIL.Image.Image):
        if isinstance(x, PIL.Image.Image):
            x = pil_to_tensor(x)
        if x.dtype in [torch.uint8, torch.int32] or x.max() > 1:
            x =  x/255
        x = x.to(torch.float32)
        if x.shape[0] == 1 and self.num_channels == 3:
            x = x.repeat(3, 1, 1)
        elif x.shape[0] == 3 and self.num_channels == 1:
            x = x.mean(dim=0, keepdim=True)
        return x 
        
    def __call__(self, **data) -> torch.Tensor:
        return torch.cat([self.transform(data[source]) for source in self.source], dim=0)

    
class RandomAlignedPerspective(torch.nn.Module):
    def __init__(
        self,
        distortion_range: tuple[float, float] = 0.3,
        interp: str = 'nearest-exact'
    ):
        super().__init__()
        self.distortion_range = distortion_range
        if interp == 'nearest-exact':
            self.interp = InterpolationMode.NEAREST_EXACT
        else:
            self.interp = InterpolationMode.BILINEAR

    @property
    def distortion_scale(self):
        if isinstance(self.distortion_range, tuple):
            return random.uniform(*self.distortion_range) 
        else:
            return self.distortion_range

    def forward(self, *imgs):
        img = imgs[0]
        height, width = img.shape[1:] if isinstance(img, torch.Tensor) else img.size
        d = self.distortion_scale
        start_points, end_points = RandomPerspective.get_params(width, height, d)
        x_coords = [p[0] for p in end_points]
        y_coords = [p[1] for p in end_points]
        left, top = int(min(x_coords)), int(min(y_coords))
        crop_w, crop_h = int(max(x_coords) - left), int(max(y_coords) - top)
        
        matrix = cv2.getPerspectiveTransform(np.float32(start_points), np.float32(end_points))
        inv_matrix = np.linalg.inv(matrix)
        final_imgs = []
        for img in imgs:
            if isinstance(img, torch.Tensor):
                img = to_pil_image(img)
            img_warped = perspective(img, start_points, end_points, interpolation=self.interp)
            img_cropped = crop(img_warped, top, left, crop_h, crop_w)
            final_img = resize(img_cropped, [height, width], interpolation=self.interp)
            final_imgs.append(pil_to_tensor(final_img))

        meta = {
            "inv_matrix": inv_matrix,
            "crop_rect": (left, top, crop_w, crop_h),
            "orig_size": (width, height),
            "distortion": d
        }
        
        return final_imgs, meta

    def reverse(self, img: torch.Tensor, meta: dict, exact: bool = False):
        """
        Reverses the transformation to recover the original perspective.
        Note: Quality will be degraded due to double-interpolation.
        """
        mode = InterpolationMode.NEAREST if exact else InterpolationMode.BILINEAR
        inv_matrix = meta["inv_matrix"]
        left, top, crop_w, crop_h = meta["crop_rect"]
        orig_w, orig_h = meta["orig_size"]
        img = resize(img, [crop_h, crop_w], interpolation=mode)
        right_pad = orig_w - (left + crop_w)
        bottom_pad = orig_h - (top + crop_h)
        img = pad(img, [left, top, right_pad, bottom_pad])
        start_points = np.float32([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]])
        reversed_points = self._apply_matrix_to_points(start_points, inv_matrix)
        return perspective(img, start_points.tolist(), reversed_points.tolist(), interpolation=mode)
    
    def soft_reverse(self, img: torch.Tensor, meta: dict) -> torch.Tensor:
        return self.reverse(img, meta, exact=False)
    
    def _apply_matrix_to_points(self, points, matrix):
        """Helper to transform coordinates using a 3x3 matrix."""
        points_homo = np.concatenate([points, np.ones((4, 1))], axis=1)
        transformed = points_homo @ matrix.T
        return transformed[:, :2] / transformed[:, 2:]
    
