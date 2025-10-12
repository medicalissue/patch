from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


class LocalImageDataset(Dataset):
    """Dataset for loading images from a local directory."""

    def __init__(self, folder_path: str, transform: Optional[Callable[[Image.Image], Tensor]] = None):
        self.folder_path = Path(folder_path)
        self.transform = transform

        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPEG", ".JPG", ".PNG"}
        self.image_paths = sorted(
            f for f in self.folder_path.iterdir()
            if f.is_file() and f.suffix in valid_extensions
        )

        if not self.image_paths:
            raise ValueError(f"No images found in {folder_path}")

        print(f"  Found {len(self.image_paths)} images in {folder_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_path.name
