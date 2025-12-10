import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class DefectDataset(Dataset):
    def __init__(self, root_dir_list, class_map=None, transform=None):
        self.samples = []
        self.transform = transform
        self.class_map = class_map or {}
        class_counter = 0

        for folder in root_dir_list:
            defect_type = os.path.basename(folder).split("-")[0]
            if defect_type not in self.class_map:
                self.class_map[defect_type] = class_counter
                class_counter += 1
            label = self.class_map[defect_type]
            image_folder = os.path.join(folder, "cwt_images")

            if not os.path.isdir(image_folder):
                print(f"[Warning] Folder not found: {image_folder}")
                continue

            file_list = sorted(os.listdir(image_folder), key=lambda x: int(''.join(filter(str.isdigit, x))))
            for fname in file_list:
                if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.samples.append((os.path.join(image_folder, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_custom_dataloader(base_dir, domain_tag_list, batch_size=32, shuffle=True):
    paths = []
    for name in domain_tag_list:
        full_path = os.path.join(base_dir, name)
        print(f"[{name}] loading from:", os.path.join(full_path, "cwt_images"))
        paths.append(full_path)

    transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = DefectDataset(paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=True)
    return dataloader


def print_samples_per_label_per_domain(loader_name, loader):
    print(f"\n===> Sample preview for: {loader_name}")
    dataset = loader.dataset

    label_domain_samples = defaultdict(dict)

    for img_path, label in dataset.samples:
        parts = img_path.split(os.sep)
        domain = parts[1]
        if domain not in label_domain_samples[label]:
            label_domain_samples[label][domain] = img_path

    for label in sorted(label_domain_samples.keys()):
        print(f"Label {label}:")
        for domain, path in label_domain_samples[label].items():
            print(f"  {path}")
