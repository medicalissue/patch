"""
Grid-level metrics for patch detection

Computes grid cell intersection with patch polygons and calculates
detection metrics at the spatial resolution level.
"""

import numpy as np
from typing import List, Tuple, Dict


def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm

    Args:
        x, y: Point coordinates
        polygon: List of (x, y) tuples forming the polygon

    Returns:
        True if point is inside polygon
    """
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def line_segment_intersects_rect(x1: float, y1: float, x2: float, y2: float,
                                   rect_x: float, rect_y: float,
                                   rect_w: float, rect_h: float) -> bool:
    """
    Check if a line segment intersects with a rectangle

    Args:
        x1, y1, x2, y2: Line segment endpoints
        rect_x, rect_y: Rectangle top-left corner
        rect_w, rect_h: Rectangle width and height

    Returns:
        True if line segment intersects rectangle
    """
    # Check if either endpoint is inside rectangle
    if (rect_x <= x1 <= rect_x + rect_w and rect_y <= y1 <= rect_y + rect_h):
        return True
    if (rect_x <= x2 <= rect_x + rect_w and rect_y <= y2 <= rect_y + rect_h):
        return True

    # Check intersection with rectangle edges
    rect_edges = [
        (rect_x, rect_y, rect_x + rect_w, rect_y),  # Top
        (rect_x + rect_w, rect_y, rect_x + rect_w, rect_y + rect_h),  # Right
        (rect_x, rect_y + rect_h, rect_x + rect_w, rect_y + rect_h),  # Bottom
        (rect_x, rect_y, rect_x, rect_y + rect_h),  # Left
    ]

    for ex1, ey1, ex2, ey2 in rect_edges:
        if segments_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
            return True

    return False


def segments_intersect(x1: float, y1: float, x2: float, y2: float,
                       x3: float, y3: float, x4: float, y4: float) -> bool:
    """Check if two line segments intersect"""
    def ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) > (by - ay) * (cx - ax)

    return (ccw(x1, y1, x3, y3, x4, y4) != ccw(x2, y2, x3, y3, x4, y4) and
            ccw(x1, y1, x2, y2, x3, y3) != ccw(x1, y1, x2, y2, x4, y4))


def polygon_intersects_cell(polygon: List[Tuple[float, float]],
                            cell_x: float, cell_y: float,
                            cell_size: float, image_size: int = 224) -> bool:
    """
    Check if a polygon intersects with a grid cell (even if only corners touch)

    Args:
        polygon: List of (x, y) tuples in 224x224 space
        cell_x, cell_y: Grid cell indices (0-based)
        cell_size: Size of each cell in pixels
        image_size: Image size (default 224)

    Returns:
        True if polygon touches or overlaps the cell
    """
    if not polygon or len(polygon) < 3:
        return False

    # Calculate cell boundaries in pixel space
    x_start = cell_x * cell_size
    y_start = cell_y * cell_size
    x_end = min((cell_x + 1) * cell_size, image_size)
    y_end = min((cell_y + 1) * cell_size, image_size)

    # Check if any polygon vertex is inside the cell
    for px, py in polygon:
        if x_start <= px <= x_end and y_start <= py <= y_end:
            return True

    # Check if any cell corner is inside the polygon
    cell_corners = [
        (x_start, y_start),
        (x_end, y_start),
        (x_end, y_end),
        (x_start, y_end)
    ]
    for cx, cy in cell_corners:
        if point_in_polygon(cx, cy, polygon):
            return True

    # Check if any polygon edge intersects with the cell
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if line_segment_intersects_rect(x1, y1, x2, y2, x_start, y_start,
                                       x_end - x_start, y_end - y_start):
            return True

    return False


def compute_ground_truth_grid(patch_corners: List[Dict],
                              spatial_resolution: int,
                              image_size: int = 224) -> np.ndarray:
    """
    Compute ground truth grid mask from patch corners

    Args:
        patch_corners: List of dicts with 'x' and 'y' keys (4 corners)
        spatial_resolution: Number of grid cells per side (e.g., 14 for 14x14)
        image_size: Image size in pixels (default 224)

    Returns:
        Binary mask [spatial_resolution, spatial_resolution] where 1 = patch cell
    """
    gt_grid = np.zeros((spatial_resolution, spatial_resolution), dtype=bool)

    if not patch_corners or len(patch_corners) < 4:
        return gt_grid

    # Convert corners to polygon
    polygon = [(c['x'], c['y']) for c in patch_corners]

    # Calculate cell size
    cell_size = image_size / spatial_resolution

    # Check each grid cell
    for i in range(spatial_resolution):
        for j in range(spatial_resolution):
            if polygon_intersects_cell(polygon, j, i, cell_size, image_size):
                gt_grid[i, j] = True

    return gt_grid


def compute_grid_metrics(gt_grid: np.ndarray, pred_grid: np.ndarray) -> Dict[str, float]:
    """
    Compute detection metrics at grid level

    Args:
        gt_grid: Ground truth binary mask [H, W]
        pred_grid: Predicted binary mask [H, W]

    Returns:
        Dictionary with metrics: tp, tn, fp, fn, accuracy, precision, recall, f1
    """
    gt_flat = gt_grid.flatten().astype(bool)
    pred_flat = pred_grid.flatten().astype(bool)

    tp = int(np.logical_and(pred_flat, gt_flat).sum())
    tn = int(np.logical_and(~pred_flat, ~gt_flat).sum())
    fp = int(np.logical_and(pred_flat, ~gt_flat).sum())
    fn = int(np.logical_and(~pred_flat, gt_flat).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_cells': total
    }
