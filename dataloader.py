from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class LocalImageDataset(Dataset):
    """로컬 폴더의 이미지를 로드하는 Dataset"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = Path(folder_path)
        self.transform = transform
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.JPEG', '.JPG', '.PNG'}
        
        self.image_paths = [
            f for f in self.folder_path.iterdir() 
            if f.suffix in valid_extensions
        ]
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {folder_path}")
        
        print(f"  Found {len(self.image_paths)} images in {folder_path}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path.name)