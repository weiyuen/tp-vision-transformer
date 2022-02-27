import pytorch_lightning as pl
import torch
import torchmetrics
import torchvision

from imgaug import augmenters as iaa
from torch import nn
from torch.utils.data import DataLoader

from tp_vit import TPViT


class PLModel(pl.LightningModule):
    def __init__(self, lr=1e-4, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = TPViT(**kwargs)
        self.loss = nn.modules.loss.CrossEntropyLoss()
        self.acc = torchmetrics.Accuracy()
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        acc = self.acc(outputs, y)
        self.log('train_loss', loss)
        self.log('train_acc', acc, on_step=True, on_epoch=False)
        return loss
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        acc = self.acc(outputs, y)
        self.log('val_loss', loss)
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        outputs = self.forward(x)
        loss = self.loss(outputs, y)
        acc = self.acc(outputs, y)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9
        )


def main():
    # model parameters
    n_classes = 1000
    dropout = 0.1

    # training parameters
    batch_size = 64
    epochs = 200
    lr = 5e-6


    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.2),
        torchvision.transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), shear=5),
        torchvision.transforms.ToTensor()
    ])
    
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor()
    ])

    train_ds = torchvision.datasets.ImageFolder(r'B:\Datasets\ImageNet2\train', transform=transform)
    valid_ds = torchvision.datasets.ImageFolder(r'B:\Datasets\ImageNet2\validation', transform=val_transform)

    train_datagen = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=12)
    valid_datagen = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=12)
    model = PLModel(lr=lr, n_classes=n_classes, dropout=dropout, ff_dropout=dropout)
    model = PLModel.load_from_checkpoint(r'lightning_logs\version_30\checkpoints\epoch=164-step=3303134.ckpt')
    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')

    trainer = pl.Trainer(gpus=1, max_epochs=epochs, callbacks=ckpt_callback, auto_lr_find=True)
    trainer.fit(model, train_datagen, valid_datagen)

if __name__ == '__main__':
    main()
