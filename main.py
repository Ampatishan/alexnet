import os
import torch
import torchvision
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from AlexNet import AlexNet

model = AlexNet()
model = model.to("cuda")

transform = transforms.Compose([T.RandomResizedCrop(224), T.ToTensor()])


train_dataset = torchvision.datasets.ImageNet("./", split="train", transform=transform)
val_dataset = torchvision.datasets.ImageNet("./", split="val", transform=transform)


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for i in range(1000):
    running_loss = 0.0
    steps = 0
    model.train()
    for image, label in tqdm(train_loader):
        image = image.to("cuda")
        label = label.to("cuda")

        optimizer.zero_grad()

        prediction = model(image)
        loss = loss_fn(prediction, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        steps += 1
    tr_loss = running_loss / steps
    val_loss = 0.0
    val_steps = 0
    for val_image, val_label in val_loader:
        val_image = val_image.to("cuda")
        val_label = val_label.to("cuda")

        val_pred = model(val_image)
        loss = loss_fn(val_pred, val_label)

        val_loss += loss.item()
        val_steps += 1
    val_loss = val_loss / val_steps

    # wandb.log({"epoch": i, "loss": tr_loss,"val_loss":val_loss})
    print("loss : {},   val_loss : {}".format(tr_loss, val_loss))
    # scheduler.step(tr_loss)

print("Finished Training")
