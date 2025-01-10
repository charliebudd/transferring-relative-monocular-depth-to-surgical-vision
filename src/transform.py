import torch
from torchvision.transforms import Compose, ColorJitter, RandomGrayscale
from torchvision.transforms import RandomHorizontalFlip, RandomAffine, InterpolationMode

class StrongTransform():
    def __init__(self, do_spatial=True):
        self.do_spatial = do_spatial
        self.photometric_transform = Compose([
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            RandomGrayscale(p=0.1),
        ])
        self.spatial_transform = Compose([
            RandomHorizontalFlip(),
            RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15, interpolation=InterpolationMode.BILINEAR),
        ])
        
    def __call__(self, rgb_images, depth=None):
        
        if depth == None:
            images = []
            for image in rgb_images:
                image = self.photometric_transform(image)
                if self.do_spatial:
                    image = self.spatial_transform(image)
                images.append(image)
            images = torch.stack(images)
            return images
        else:
            images = []
            depths = []
            for image, depth in zip(rgb_images, depth):
                image = self.photometric_transform(image)
                if self.do_spatial:
                    stack = torch.cat([image, depth], dim=0)
                    stack = self.spatial_transform(stack)
                    image, depth = stack[:3], stack[3:]
                images.append(image)
                depths.append(depth)
            images = torch.stack(images)
            depths = torch.stack(depths)
            return images, depths