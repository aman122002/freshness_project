import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class FruitVegDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to 'data/raw' folder
        transform: torchvision transforms to apply
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        for fruit_folder in os.listdir(root_dir):
            fruit_path = os.path.join(root_dir, fruit_folder)
            if not os.path.isdir(fruit_path):
                continue

            # Determine label from folder name
            if "_healthy" in fruit_folder.lower():
                label = 1
            elif "_rotten" in fruit_folder.lower():
                label = 0
            else:
                continue  # skip unexpected folders

            # Add images
            for img_name in os.listdir(fruit_path):
                if img_name.endswith((".jpg", ".png", ".jpeg")):
                    self.images.append(os.path.join(fruit_path, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# Example usage
if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = FruitVegDataset(root_dir="../data/v2", transform=transform)
    print(f"Total images: {len(dataset)}")
    img, label = dataset[0]
    print(f"Image shape: {img.shape}, Label: {label}")
