import os

from datasets import load_dataset
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

class CelebA(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.dataset = load_dataset("student/celebA")['train']

    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, idx):
        entry = self.dataset[idx]
        image, label = entry['image'], entry['label']
        image = transforms.ToTensor()(image)
        label = torch.tensor(label)
        return image, label


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dataloader = DataLoader(CelebA(), batch_size=512, shuffle=True)
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-3)
    for _ in tqdm(range(100)):
        for is_train, _dataloader in zip([True, False], [dataloader, dataloader]):
            for (image, label) in tqdm(_dataloader):

                if is_train:
                    optimizer.zero_grad()
                    prediction = model(image.cuda())
                    argmax = torch.argmax(prediction, 1)
                    loss = criterion(prediction, label.cuda())
                    loss.backward()
                else:
                    model(image.cuda())
