import os
import pathlib
import sys

import torch
import torchvision
import time
# Configuration
config_batch_size   = 250
config_dataset_root = pathlib.Path("/tmp")
config_dataset_name = "CIFAR10"
config_device       = "cuda" if torch.cuda.is_available() else "cpu"
config_nb_workers   = len(os.sched_getaffinity(0))
config_train_lr     = 0.2
config_train_mt     = 0.9
config_train_wd     = 0.0005

# Define the transformations
transforms_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

read_t = time.time()
transforms_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
print("time to read is: ", time.time() - read_t)
# Load the dataset
dataset_data_path   = str(config_dataset_root / config_dataset_name)
dataset_data_loader = getattr(torchvision.datasets, config_dataset_name)
dataset_data_train  = dataset_data_loader(dataset_data_path, train=True, download=True, transform=transforms_train)
dataset_data_test   = dataset_data_loader(dataset_data_path, train=False, download=False, transform=transforms_test)
dataset_train       = torch.utils.data.DataLoader(dataset_data_train, batch_size=config_batch_size, shuffle=True, num_workers=max(1, (config_nb_workers + 1) // 2))
dataset_test        = torch.utils.data.DataLoader(dataset_data_test, batch_size=len(dataset_data_test), shuffle=False, num_workers=max(1, config_nb_workers // 2))

# Build the model
model = torchvision.models.resnet50()
model = model.to(config_device)
if config_device == "cuda":
  model = torch.nn.DataParallel(model)

# Define the loss, optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config_train_lr, momentum=config_train_mt, weight_decay=config_train_wd)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50], gamma=0.1)

# Train and test the model
def train(epoch):
  sys.stdout.write("Training for epoch " + str(epoch) + "...")
  sys.stdout.flush()
  try:
    model.train()
    sum_loss = 0
    for inputs, targets in dataset_train:
      inputs, targets = inputs.to(config_device), targets.to(config_device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, targets)
      loss.backward()
      optimizer.step()
      sum_loss += loss.item()
    print("epoch: ", epoch, "loss: ", sum_loss)
  except:
    print(" failed.")
    raise
  print(" " + str(sum_loss))

def test(epoch):
  total_inf = 0
  sys.stdout.write("Testing  for epoch " + str(epoch) + "...")
  sys.stdout.flush()
  try:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
      for inputs, targets in dataset_test:
        inputs, targets = inputs.to(config_device), targets.to(config_device)
        inf_t = time.time()
        outputs = model(inputs)
        total_inf+= time.time() - inf_t
        _, predicted = outputs.max(1)
        total   += targets.size(0)
        correct += predicted.eq(targets).sum().item()
  except:
    print(" failed.")
    raise
  print(" " + str(correct * 100 / total) + " %")
  print("total inference time:", total_inf)

try:
  epoch = 0
  scheduler.step()
  while True:
    epoch += 1
    scheduler.step()
    train(epoch)
    test(epoch)
except KeyboardInterrupt:
  pass
