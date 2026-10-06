from __future__ import annotations
import xml.etree.ElementTree as ET
import torch, re, math, random
import shapely
ET.register_namespace('', "http://www.w3.org/2000/svg")


class Polygon:

    def __init__(self, points: torch.Tensor, color: str):
        self.points = points.float()
        self.color = color
        
    def __repr__(self):
        return f'Polygon({self.TYPE}, n={len(self.points)}, color={self.color}, points={self.points}, sides={self.sides.tolist()})'
        
    def copy(self) -> Polygon:
        return Polygon(self.points.clone(), self.color)
    
    def __eq__(self, other: Polygon) -> bool:
        if self.points.shape != other.points.shape:
            return False
        return (self.points - other.points).abs().mean() <self.area*1e-3

    def dist(self, other: Polygon) -> float:
        """Compute the distance between other Polygon."""
        return torch.linalg.norm(self.center - other.center)
          
    @property
    def left(self) -> float:
        return self.points[:, 0].min().item()
    
    @property
    def right(self) -> float:
        return self.points[:, 1].max().item()
    
    @property
    def top(self) -> float:
        return self.points[:, 1].max().item()
    
    @property
    def bottom(self) -> float:
        return self.points[:, 1].min().item()
        
    @property
    def num_sides(self) -> int:
        return self.points.shape[0]
    
    def is_triangle(self) -> bool:
        if self.num_sides != 3:
            return False
        sides = self.sides.sort().values
        min_value = sides.max()*0.15
        return (sides[1] - sides[0] < min_value and abs((sides[:2]**2).sum().sqrt() - sides[2]) < min_value)
    
    @property
    def squared_sides(self) -> torch.Tensor:
        """Calculates the squared length of all sides of the polygon."""
        vectors = self.points - torch.roll(self.points, shifts=(-1,), dims=0)
        return torch.sum(vectors ** 2, dim=1)
    
    @property
    def sides(self) -> torch.Tensor:
        return torch.sqrt(self.squared_sides)

    def overlap(self, other: Polygon) -> float:
        """Measures the overlap against other polygon."""
        p1 = shapely.Polygon(self.points.numpy())
        p2 = shapely.Polygon(other.points.numpy())
        if not (p1.is_valid and p2.is_valid):
            return 0
        intersection_area = p1.intersection(p2).area
        union_area = p1.union(p2).area
        if union_area == 0:
            return union_area
        return intersection_area/union_area
    
    @property 
    def area(self) -> float:
        X = self.points[:, 0]
        Y = self.points[:, 1]
        X_next = torch.roll(X, shifts=(-1,), dims=0)
        Y_next = torch.roll(Y, shifts=(-1,), dims=0)
        term1 = X * Y_next
        term2 = X_next * Y
        sum_of_crosses = torch.sum(term1 - term2)
        area = 0.5 * torch.abs(sum_of_crosses)
        return area.item()
    
    @property
    def TYPE(self) -> str:
        if self.is_triangle():
            return 'triangle'
        elif self.is_square():
            return 'square'
        elif self.is_parallelogram():
            return 'parallelogram'
        else:
            return 'unknown'
        
    @property
    def perimeter(self) -> float:
        return self.sides.sum()
        
    @property
    def center(self) -> torch.Tensor:
        """Calculates the geometric center (centroid) of the untransformed polygon."""
        return self.points.mean(0)
    
    def is_equilateral(self) -> bool:
        """Checks if all sides have equal length (within tolerance)."""
        sides = self.sides
        return (torch.max(sides).int() - torch.min(sides).int())/sides.sum() < 0.05
    
    def is_parallelogram(self) -> bool:
        sides = self.sides
        if len(sides) != 4:
            return False 
        points = self.points 
        
        # check if opposite sides are equal in length
        if abs(sides[0] - sides[2])/sides.sum() > 0.05:
            return False 
        if abs(sides[1] - sides[3])/sides.sum() > 0.05:
            return False 
        
        # opposite sides are parallel 
        cross2d = lambda u, v: (u[0] * v[1] - u[1] * v[0]).abs().item()
        edges = points - torch.roll(points, shifts=-1, dims=0)  # shape (4, 2)
        parallel_01 = cross2d(edges[0], edges[2]) / (sides[0] * sides[2]).item()
        parallel_12 = cross2d(edges[1], edges[3]) / (sides[1] * sides[3]).item()
        angle_tol = 0.15  # ~sin(3°), tight enough for Tangram pieces
        return parallel_01 < angle_tol and parallel_12 < angle_tol

    
    def is_square(self, tolerance: float = 0.2) -> bool:
        if len(self.sides) != 4:
            return False 
        if not self.is_equilateral():
            return False 
        edges = self.points - torch.roll(self.points, shifts=-1, dims=0)  # (4, 2)
        next_edges = torch.roll(edges, shifts=-1, dims=0)
        dots = (edges * next_edges).sum(dim=1)                  # (4,)
        norms = edges.norm(dim=1) * next_edges.norm(dim=1)      # (4,)
        cos_angles = dots / norms                               # cos(θ) ≈ 0 for right angles
        return (cos_angles.abs() < tolerance).all().item()

       
    def project_to_axis(self, axis: torch.Tensor) -> tuple[float, float]:
        """
        Projects the transformed polygon points onto a given axis (normal vector).
        Used for Separating Axis Theorem (SAT).
        """
        # Ensure axis is normalized
        axis = axis / torch.linalg.norm(axis)
        
        # Calculate dot product (projection) for each point: P dot Axis
        projections = torch.matmul(self.points, axis)
        return projections.min().item(), projections.max().item()

    def collides_with(self, other: Polygon, space: int = 5) -> bool:
        """
        Checks for collision between this polygon and another using the
        Separating Axis Theorem (SAT). A separating axis exists if the projection 
        of the two polygons onto that axis do not overlap.
        
        Args:
            other (Polygon): Other polygon.
            space (int): Minimum space required between polygons.
            
        Returns:
            Whether the two polygons collide.
        """
        polygons = [self.points, other.points]
        
        for poly_points in polygons:
            # generate the set of axes (normals to the sides) for the current polygon
            vectors = poly_points - torch.roll(poly_points, shifts=(-1,), dims=0) 
            
            # the normal is the vector rotated 90 degrees (x, y) -> (-y, x) or (y, -x) 
            # We use (-y, x)
            axes = torch.stack([-vectors[:, 1], vectors[:, 0]], dim=1)
            
            for axis in axes:
                # project both polygons onto the axis
                min1, max1 = self.project_to_axis(axis)
                min2, max2 = other.project_to_axis(axis)
                
                # check for separation (no overlap)
                if max1 < min2-space or max2 < min1-space:
                    return False
        
        return True

    def get_bounds(self) -> tuple[float, float, float, float]:
        """
        Returns the bounding box (min_x, max_x, min_y, max_y) of the 
        transformed polygon.
        """
        min_x = self.points[:, 0].min().item()
        max_x = self.points[:, 0].max().item()
        min_y = self.points[:, 1].min().item()
        max_y = self.points[:, 1].max().item()
        return min_x, max_x, min_y, max_y

    def is_in_viewbox(self, viewbox: tuple[float, float, float, float], padding: int = 3) -> bool:
        """Checks whether the polygon is in a viewbox with an specific padding."""
        min_x_curr, max_x_curr, min_y_curr, max_y_curr = self.get_bounds()
        
        x_min_vb, y_min_vb, w_vb, h_vb = viewbox
        x_max_vb = x_min_vb + w_vb
        y_max_vb = y_min_vb + h_vb

        padded_x_min = x_min_vb + padding
        padded_x_max = x_max_vb - padding
        padded_y_min = y_min_vb + padding
        padded_y_max = y_max_vb - padding

        is_x_in_bounds = (min_x_curr >= padded_x_min) and (max_x_curr <= padded_x_max)
        is_y_in_bounds = (min_y_curr >= padded_y_min) and (max_y_curr <= padded_y_max)
            
        return is_x_in_bounds and is_y_in_bounds

    def translation_range(
        self, 
        viewbox: tuple[float, float, float, float], 
        padding: int = 3
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Computes the translation range (vertical and horizontal) allowed for 
        an specified viewbox.

        Args:
            viewbox (tuple[float, float, float, float]): Viewbox to limit the translation.
            padding (int): Polygon padding.

        Returns:
            tuple[tuple[float, float], tuple[float, float]]: x and y-range of the allowed translations.
        """
        
        min_x, max_x, min_y, max_y = self.get_bounds()
        vx, vy, vw, vh = viewbox
        
        space_xmin = vx + padding
        space_xmax = vx + vw - padding
        space_ymin = vy + padding
        space_ymax = vy + vh - padding
        
        dx_min = space_xmin - min_x
        dx_max = space_xmax - max_x 
        dy_min = space_ymin - min_y
        dy_max = space_ymax - max_y
        return (dx_min, dx_max), (dy_min, dy_max)
   
    def affine(self, a: float, b: float, c: float, d: float, e: float, f: float) -> Polygon:
        """Apply affine transformation to a Polygon. 
        
        Args:
            a, b, c, d, e, f: Values of the affine matrix.
            
        Returns:
            Polygon: New Polygon instance where points are updated.
        """
        x, y = self.points.unbind(-1)
        nx = a * x + c * y + e
        ny = b * x + d * y + f
        new_points = torch.stack([nx, ny], dim=-1)
        return Polygon(new_points, self.color)
    
    def to_svg(self):
        """
        Converts the current Polygon state into an <polygon> XML element.
        """
        points_str = " ".join(f"{x:.3f} {y:.3f}" for x, y in self.points)
        polygon_element = ET.Element("polygon")
        polygon_element.set('points', points_str)
        polygon_element.set('fill', self.color)
        if self.color in ['white', '#FFFFFF']:
            polygon_element.set('stroke', self.color) 
            polygon_element.set('strokewidth', '4')
        return polygon_element
    
    @classmethod
    def read_svg_polygons(cls, svg_content: str) -> list[Polygon]:
        """
        Parses an SVG string and returns a list of Polygon objects.
        
        Args:
            svg_content (str): String containing the SVG file content.
            
            
        Returns:
            list[Polygon]: List of Polygon objects.
        """
        polygons = []
        root = ET.fromstring(svg_content)
        for element in root.iter('{http://www.w3.org/2000/svg}polygon'):
            points_str = element.get('points', '')
            points = torch.tensor(list(map(float, re.split(r'[, ]+', points_str.strip()))))
            points = points.reshape((-1, 2))
            polygon = Polygon(points, '#FFFFFF')
            
            # check if there is a transformation matrix 
            match = element.get('transform', '').removeprefix('matrix(').removesuffix(')')
            if match:
                a, b, c, d, e, f = [float(val) for val in match.replace(',', ' ').split()]
                polygon = polygon.affine(a, b, c, d, e, f)
            polygons.append(polygon)
        return polygons
    
    @classmethod
    def from_generated(cls, points: torch.Tensor, color: str) -> Polygon:
        """Creates a Polygon from a generated view (estimated points). Filters
        those sides below a fixed size."""
        candidate = Polygon(points, color)

        sides = torch.sum((points - torch.roll(points, shifts=(-1,), dims=0))**2, dim=1).sqrt()
        mask = sides > sides.max()/2*0.8
        if not mask.all() and len(points) == 4:
            mask = sides != sides.min()
            s1 = Polygon(points[mask], color).sides.sort().values
            s2 = Polygon(points[torch.roll(mask, shifts=(1,))], color).sides.sort().values
            if s1[1]-s1[0] < s2[1]-s2[0]:
                return Polygon(points[mask], color)
            else:
                return Polygon(points[torch.roll(mask, shifts=(1,))], color)
        return Polygon(points, color=color)
        
    
def get_random_affine(
    scale: tuple[float, float] | float = 1.0, 
    rotation: tuple[float, float] | float = 0.0, 
    x_translation: tuple[float, float] | float = 0.0, 
    y_translation: tuple[float, float] | float = 0.0, 
    flip: float = 0.5
):
    """Computes a random affine matrix from specified parameters.
    
    Args:
        scale (tuple[float, float]): Scaling range.
        rotation (tuple[float, float]): Rotation range.
        flip (float): Flipping probability.
        
    Returns:
        Positions of the first two rows of the affine matrix.
    """
    # random scale
    s = random.uniform(*scale) if isinstance(scale, tuple) else scale
    
    # random flipping
    fx = -1 if random.random() < flip else 1
    fy = -1 if random.random() < flip else 1
    
    # combined scale and flip matrix factors
    sx, sy = s * fx, s * fy
    
    # random rotation
    angle_deg = random.uniform(*rotation) if isinstance(rotation, tuple) else rotation
    theta = math.radians(angle_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    
    a = sx * cos_t
    b = sx * sin_t
    c = sy * -sin_t
    d = sy * cos_t
    e = random.uniform(*x_translation) if isinstance(x_translation, tuple) else x_translation
    f = random.uniform(*y_translation) if isinstance(y_translation, tuple)  else y_translation
    return (a, b, c, d, e, f)