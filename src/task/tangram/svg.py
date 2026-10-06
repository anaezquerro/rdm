from __future__ import annotations
from .polygon import Polygon, get_random_affine
import torch, PIL, io, cairosvg, cv2
import xml.etree.ElementTree as ET
import numpy as np 

from src.util import shuffle, avg, hex_to_numpy
ET.register_namespace('', "http://www.w3.org/2000/svg")

class SVG:
    """Implementation of a SVG object."""

    def __init__(self, polygons: list[Polygon], viewbox: tuple[float, float, float, float] = None):
        """
        Initialization of a SVG instance, defined by a list of polygons (ordered by type, 
        area and coordinates) and a viewbox.
        """
        self.polygons = sorted(polygons, key=lambda p: (p.TYPE, p.area))
        self.points = torch.cat([p.points for p in self.polygons], dim=0)
        if viewbox is None:
            self.fit_viewbox()
        else:
            self.viewbox = viewbox

    def __iter__(self):
        return iter(self.polygons)
        
    def __len__(self) -> int:
        return len(self.polygons)
    
    @property
    def content(self) -> dict:
        return dict(points=[p.points for p in self.polygons], colors=[p.color for p in self.polygons], viewbox=self.viewbox)
    
    @classmethod
    def from_content(cls, content: dict[str, torch.Tensor]):
        polygons = [Polygon(points, color) for points, color in zip(content['points'], content['colors'], strict=True)]
        return cls(polygons, content['viewbox'])

    def fit_viewbox(self, padding: float = 20) -> SVG:
        """Modifies the viewbox to make it square and with polygons centered."""
        width = self.right - self.left 
        height = self.bottom - self.top
        
        # size of the viewbox (both height and width)
        size = max(width, height) + padding
        
        # compute margins 
        left = self.left - (size-width)/2
        top = self.top - (size-height)/2
        self.viewbox = (left, top, size, size)
        return self
        
    @property
    def area(self) -> float:
        return float(self.viewbox[-1]*self.viewbox[-2])
    
    @property
    def left(self) -> float:
        return self.points[:, 0].min()
    
    @property
    def right(self) -> float:
        return self.points[:, 0].max()
    
    @property
    def top(self) -> float:
        return self.points[:, 1].min()
    
    @property
    def bottom(self) -> float:
        return self.points[:, 1].max()

    @property
    def colors(self) -> set[colors]:
        return [p.color for p in self]
    
    @property
    def centers(self) -> torch.Tensor:
        polygons = self.polygons[:2] + [self.polygons[4]] + \
            sorted(self.polygons[2:4], key=lambda p: p.left) + \
            sorted(self.polygons[-2:], key=lambda p: p.left)
        centers = torch.stack([p.center for p in polygons])
        centers[:, 0] /= self.viewbox[-1]
        centers[:, 1] /= self.viewbox[-2]
        return centers

    @property
    def dist(self) -> torch.Tensor:
        centers = self.centers 
        return torch.cdist(centers, centers, p=2).abs()
    
    def copy(self) -> SVG:
        return self.__class__([p.copy() for p in self.polygons], self.viewbox)
    
    def random_affine(
        self,
        rotation: tuple[float, float] = (0, 90),
    ) -> SVG:
        """Applies a random affine transformation to all polygons.
        
        Args:
            scale (tuple[float, float]): Scaling range.
            rotation (tuple[float, float]): Rotation range.
            
        Returns:
            Tangram: New Tangram instance.
        """
        matrix = get_random_affine(scale=1.0, rotation=rotation)
        new_polygons = []
        for polygon in self:
            new_polygons.append(polygon.affine(*matrix))
        new = self.__class__(new_polygons)
        return new
    
    def scale(self, ratio: float) -> SVG:
        """Modifies the viewbox to scale the SVG view."""
        new = self.copy()
        left, top, height, width = self.viewbox 
        new.viewbox = left-width/2*ratio, top-height/2*ratio, height+height*ratio, width+width*ratio
        return new 
    
    def coloring(self, colors: list[str]) -> SVG:
        """Changes the polygon colors from a given list.
        
        Args:
            colors (list[str]): List to extract colors.
            
        Returns:
            Tangram: Colored Tangram.
        """
        new = self.copy()
        for polygon, color in zip(new.polygons, colors):
            polygon.color = color
        return new
    
    def shuffle(self, padding: int = 5, attempts: int = 1000) -> SVG:
        """
        Shuffles the polygons by applying random rotations, flips, and 
        translations such that they do not overlap and remain within the viewbox.
        The function is used to create the shuffle views for the reasoning task.
        """
        placed_polygons = []
        for polygon in sorted(self.polygons, key=lambda p: p.area, reverse=True):
            attempt = attempts
            placed = False
            x_range, y_range = polygon.translation_range(self.viewbox, padding)
            
            while attempt > 0 and not placed:
                candidate = polygon.affine(*get_random_affine(rotation=(-45, 45), x_translation=x_range, y_translation=y_range))
                if candidate.is_in_viewbox(self.viewbox, padding) and \
                    not any(candidate.collides_with(p, padding) for p in placed_polygons):
                    placed_polygons.append(candidate)
                    placed = True 
                attempt -= 1
            if not placed:
                return self.scale(ratio=0.9)
        new = self.__class__(placed_polygons, self.viewbox)
        return new
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_svg())
            
    @classmethod
    def load(cls, path: str) -> SVG:
        with open(path, 'r', encoding='utf-8') as f:
            svg_content = f.read()
        polygons = Polygon.read_svg_polygons(svg_content)
        assert len(polygons) > 0, f'Error in {path}: no polygons found!'
        svg = cls(polygons)
        return svg
    
    def to_img(self) -> PIL.Image:
        png_data = cairosvg.svg2png(
            bytestring=self.to_svg().encode('utf-8'),
            output_width=512,
            output_height=512
        )
        data_stream = io.BytesIO(png_data)
        img = PIL.Image.open(data_stream).convert('RGBA')
        
        # Create a solid background to remove transparency artifacts
        background = PIL.Image.new("RGBA", img.size, (0, 0, 0, 255))
        composite = PIL.Image.alpha_composite(background, img)
        return composite.convert('RGB')
    
    def to_svg(self) -> str:
        root = ET.Element("svg", viewBox=' '.join(f'{v:.6f}' for v in self.viewbox), xmlns="http://www.w3.org/2000/svg")
        for polygon in self.polygons:
            root.append(polygon.to_svg())
        tree = ET.ElementTree(root)
        svg_content = ET.tostring(root, encoding='utf-8').decode('utf-8')
        svg_content = svg_content.replace('>', '>\n').replace('<polygon', '\t<polygon')
        return '<?xml version="1.0" encoding="utf-8"?>\n' + svg_content
    
    def is_tangram(self) -> bool:
        """Checks whether the SVG object is a Tangram (7 valid tans)"""
        if len(self) != 7:
            return False
        if sum(p.TYPE == 'square' for p in self) != 1:
            return False 
        if sum(p.TYPE == 'parallelogram' for p in self) != 1:
            return False
        if any(p.TYPE == 'unknown' for p in self):
            return False
        # check sizes 
        square, prl, st1, st2, mt, lt1, lt2 = [round(p.area) for p in self]
        sizes = [square, prl, st1+st2, mt]
        if (max(sizes)-min(sizes))/max(sizes) > 0.5:
            return False 
        if abs(st1-st2)/max(st1, st2) > 0.5:
            return False 
        if abs(lt1-lt2)/max(lt1, lt2) > 0.5:
            return False
        if abs(1-avg(sizes)*2/avg([lt1, lt2])) > 0.5:
            return False
        return True 

    @classmethod
    def from_generated(
        cls,
        img: np.array,
        hex_colors: list[str],
    ):
        """Load an SVG from a predicted image. It assumes that each polygon 
        is colored with a different color. It requires the quantized image and colors.
        """
        # quantize 
        pixels = img.reshape(-1, 3)
        palette = hex_to_numpy(list(hex_colors) + ['#000000']).astype(int)
        dist = np.sqrt(np.sum((pixels[:, np.newaxis, :] - palette[np.newaxis, :, :])**2, axis=2))
        qimg = palette[np.argmin(dist, axis=1)].reshape(img.shape).astype(np.uint8)

        # search polygons
        polygons = []
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        for color, rgb_color in zip(hex_colors, hex_to_numpy(hex_colors)):
            mask = cv2.inRange(qimg, rgb_color, rgb_color)

            # apply erosion and dilation to remove small regions
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue 
                peri = cv2.arcLength(contour, True)
                poly = lambda e: torch.from_numpy(cv2.approxPolyDP(contour, e, True).reshape(-1,2))
                eps = 0.01 * peri
                candidate = Polygon(poly(eps), color)
                while candidate.num_sides > 2 and candidate.TYPE == 'unknown':
                    eps += 1 
                    candidate = Polygon(poly(eps), color)
                if candidate.TYPE == 'unknown':
                    continue 
                polygons.append(candidate)
        return SVG(polygons)  if len(polygons) > 0 else None

    def remove_overlap(self, ratio: 0.9, colors: list[str]) -> SVG:
        """Removes overlapping polygons with a certain overlapping ratio."""
        polygons = []
        for p in sorted(self.polygons, key=lambda p: int(p.color in colors), reverse=True):
            if not any(p.overlap(other) > ratio for other in polygons):
                polygons.append(p)
        return self.__class__(polygons, self.viewbox)

