import torch
import torchvision.transforms as transforms

norm1 = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
norm2 = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform1 = transforms.Compose([
    transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    transforms.Lambda(lambda x: x.to(torch.float32) / 255.0),
    transforms.Normalize(norm1),
    
])

transform2 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),             
    transforms.RandomHorizontalFlip(),                
    transforms.RandomRotation(15),                    
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
    transforms.ToTensor(),                            
    transforms.Normalize(norm2)
])

transform_test1 = transforms.Compose([
    transforms.Normalize(norm1),
])