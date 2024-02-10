# Applications and final projects

```{figure} images/5_applications/cover.png
---
width: 50%
name: cover
alt: cover
---
Image generated using [OpenDALL-E](https://huggingface.co/spaces/mrfakename/OpenDalleV1.1-GPU-Demo)
```

Processing images and audio data is at the core of many applications. This chapter will overview some of the most common applications in this field and how they can be implemented using the tools presented in this book.

# Computer vision

Computer vision is the field of computer science that deals with the automatic extraction of information from images. It is a very broad field that includes many different tasks:
- Image classification: assign a label to an image (e.g. which kind of plant is in a picture).
- Object detection: detect specific objects in an image (e.g. cars, pedestrians, etc.).
- Image segmentation: assign a label to each pixel of an image (e.g. which pixels belong to a car).
- Image generation: generate new images (e.g. generate a picture given a text description).
- Image captioning: generate a text description of an image (e.g. describe the content of a picture).
- ... many more!

## Image classification

Image classification is the task of assigning a label to an image. For example, given an image of a dog, the goal is to assign the label "dog" to it. Both CNNs and transformers can be used for image classification. 

**CNN implementation of image classification**

The following code shows how to implement image classification using a CNN. The code is based on the [PyTorch tutorial on image classification](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

```{code-block} python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define a CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the data
dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transforms.ToTensor())

trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

valloader = torch.utils.data.DataLoader(valset, batch_size=4,
                                            shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transforms.ToTensor()) 
            
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define the model
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(net, trainloader, criterion, optimizer):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader)):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # to GPU
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss / len(trainloader)

def evaluate(net, valloader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(valloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(valloader), correct / total

# Train the model
net.to(device)

for epoch in range(10):  # loop over the dataset multiple times

    train_loss = train_one_epoch(net, trainloader, criterion, optimizer)
    val_loss, val_acc = evaluate(net, valloader, criterion)
    print(f"Epoch {epoch} - Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f} - Val acc: {val_acc:.3f}")

print('Finished Training')

# Evaluate the model on the test set
test_loss, test_acc = evaluate(net, testloader, criterion)
print(f"Test loss: {test_loss:.3f} - Test acc: {test_acc:.3f}")
```

This code uses a standard CNN architecture for image classification. The model is trained on the CIFAR10 dataset, which contains **10 classes** of images: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck. The model achieves an accuracy of ~58% on the test set.

```{admonition} Exercise - implement a ViT transformer for image classification - 45 min
:class: exercise
Implement a ViT transformer for image classification. You can use the pre-trained ViT model from the [HuggingFace model hub](https://huggingface.co/models?pipeline_tag=image-classification) and fine-tune it on the CIFAR10 dataset.

You can use the model tag `google/vit-base-patch16-224` and the feature extractor tag `google/vit-base-patch16-224` to get the model and the feature extractor respectively.
✋ Remember that *each* model has its own feature extractor. The model documentation is available [here](https://huggingface.co/docs/transformers/model_doc/vit).
```

**Solution**

````{admonition} Solution
:class: dropdown

```{code-block} python
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from transformers import ViTFeatureExtractor, ViTForImageClassification

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ViTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        # CiFAR images are 32x32, ViT requires 224x224
        self.resize = transforms.Resize((224, 224))

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.resize(image)
        image = self.feature_extractor(images=image, return_tensors="pt")["pixel_values"]
        return {
            "pixel_values": image.squeeze(),
            "labels": label
        }

    def __len__(self):
        return len(self.dataset)

# Load the data
def get_cifar_dataloaders():
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transforms.ToTensor())

    trainset, valset = torch.utils.data.random_split(dataset, [40000, 10000])
    trainset = ViTDataset(trainset)
    valset = ViTDataset(valset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                            shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=8,
                                                shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())
    testset = ViTDataset(testset)
            
    testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                            shuffle=False, num_workers=2)
    return trainloader, valloader, testloader

trainloader, valloader, testloader = get_cifar_dataloaders()

# Define the model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=10, ignore_mismatched_sizes=True)
model.to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

def train_one_epoch(model, trainloader, criterion, optimizer):
    running_loss = 0.0
    for i, batch in enumerate(tqdm(trainloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch["pixel_values"])
        logits = outputs.logits
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss / len(trainloader)

def evaluate(model, valloader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(valloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            logits = outputs.logits
            loss = criterion(logits, labels)
            running_loss += loss.item()
            predicted = torch.argmax(logits, dim=-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(valloader), correct / total

# Train the model
for epoch in range(10):  # loop over the dataset multiple times
    train_loss = train_one_epoch(model, trainloader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, valloader, criterion)
    print(f"Epoch {epoch} - Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f} - Val acc: {val_acc:.3f}")

print('Finished Training')

# Evaluate the model on the test set
test_loss, test_acc = evaluate(model, testloader, criterion)
print(f"Test loss: {test_loss:.3f} - Test acc: {test_acc:.3f}")
```
````

# Speech Processing

Speech processing is the field of computer science that deals with the automatic extraction of information from audio signals. It is a very broad field that includes many different tasks:
- Speech recognition: convert speech to text.
- Speaker recognition: identify the speaker from a speech signal.
- Speech synthesis: generate speech from text.
- Speech translation: translate speech from one language to another.
- ... many more!

Audio signals can be analyzed using two different representations: the **time-domain** representation and the **frequency-domain** representation. The time-domain representation is the most intuitive one: it represents the amplitude of the signal as a function of time. The frequency-domain representation is obtained by applying a Fourier transform to the time-domain representation. It represents the amplitude of the signal as a function of frequency. 

**Time-frequency representations** are usually treated as images and can be processed using CNNs or transformers. **Time-domain representations**, on the other hand, are time series and can be processed using RNNs or transformers.

## Keyword spotting

Keyword spotting is the task of detecting specific words in an audio signal. For example, given an audio signal, the goal is to detect the word "yes" in it. Both CNNs and transformers can be used for the task. One practical application of keyword spotting is the detection of wake words in smart speakers. For example, the wake word "Alexa" is used to activate the Amazon Echo smart speaker.

```{admonition} Exercise - implement a transformer for keyword spotting - 45 min
:class: exercise

Using the [superb](https://huggingface.co/superb) dataset, implement a transformer for keyword spotting. You can use the pre-trained transformer from the [HuggingFace model hub](https://huggingface.co/models?pipeline_tag=audio-classification) and fine-tune it on the superb dataset. Alternatively, you can implement your own transformer-based model from scratch using the [PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) implementation of encoder layers.
```

**Solution**

````{admonition} Solution
:class: dropdown

```{code-block} python

# load dataset
from datasets import load_dataset
train_dataset = load_dataset("superb", "ks", split="train")
val_dataset = load_dataset("superb", "ks", split="validation")
test_dataset = load_dataset("superb", "ks", split="test")

print(train_dataset[0])
print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(val_dataset)}")
print(f"Number of testing examples: {len(test_dataset)}")

# Define the model
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
unique_labels = set([example["label"] for example in train_dataset])
num_labels = len(unique_labels)
model = AutoModelForAudioClassification.from_pretrained("microsoft/wavlm-base-plus", num_labels=num_labels)
print(f"Initalized model with {num_labels} labels")


# implement the dataset class
import torch

class KSDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        self.max_length_in_seconds = 2

    def __getitem__(self, idx):
        audio_array = self.dataset[idx]["audio"]["array"]
        label = self.dataset[idx]["label"]
        audio = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            max_length=self.max_length_in_seconds * 16000,
            padding="max_length",
            return_tensors="pt",
            truncation=True
        ).input_values
        return {
            "input_values": audio.squeeze(),
            "labels": label
        }

    def __len__(self):
        return len(self.dataset)
    
train_ds = KSDataset(train_dataset)
val_ds = KSDataset(val_dataset)
test_ds = KSDataset(test_dataset)

from torch import nn, optim
from tqdm.notebook import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data loaders
trainloader = torch.utils.data.DataLoader(train_ds, batch_size=16,
                                            shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(val_ds, batch_size=16,
                                            shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=16,
                                            shuffle=False, num_workers=2)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
lamda_fn = lambda epoch: 0.95 ** epoch
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lamda_fn)

def train_one_epoch(model, trainloader, criterion, optimizer):
    running_loss = 0.0
    for i, batch in enumerate(tqdm(trainloader)):
        batch = {k: v.to(device) for k, v in batch.items()}

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(batch["input_values"])
        logits = outputs.logits
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
    return running_loss / len(trainloader)

def evaluate(model, valloader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(valloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch["input_values"])
            labels = batch["labels"]
            logits = outputs.logits
            loss = criterion(logits, labels)
            running_loss += loss.item()
            predicted = torch.argmax(logits, dim=-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / len(valloader), correct / total

# Train the model
model.to(device)
best_model = None
best_acc = 0.0
for epoch in range(10):  # loop over the dataset multiple times
    train_loss = train_one_epoch(model, trainloader, criterion, optimizer)
    val_loss, val_acc = evaluate(model, valloader, criterion)
    if val_acc > best_acc:
        best_acc = val_acc
        best_model = model.state_dict()
    print(f"Epoch {epoch} - Train loss: {train_loss:.3f} - Val loss: {val_loss:.3f} - Val acc: {val_acc:.3f}")
    scheduler.step()
    
print('Finished Training')

# load the best model
model.load_state_dict(best_model)

# Evaluate the model on the test set
test_loss, test_acc = evaluate(model, testloader, criterion)
print(f"Test loss: {test_loss:.3f} - Test acc: {test_acc:.3f}")
```
````


# Conclusion

In this chapter, we have seen how to use CNNs and transformers for image and audio processing. We have seen how to implement a CNN for image classification and how to implement a transformer for keyword spotting. We have also seen how to use pre-trained models for these tasks.

<!-- 
```{admonition} Final project - 1 week
As a final assignment for the course, you are asked to provide a project that uses the tools presented in this course. You can choose any topic and dataset you like, as long as it is related to image or audio processing. You are free to choose the model you want to use (CNN, transformer, etc.). You can use the code from the previous exercises as a starting point.

Create a new repository on GitHub and upload your code there. Structure your code in a way that is easy to understand and to use. 
Within 1 week from the end of the course, you will be asked to submit a report describing your project. The report should include:
- A description of the dataset and the task.
- A brief overview of the model you used.
- A discussion of the results you obtained.
- A summary of the lessons learned and the challenges encountered.
- If relevant, a discussion of the next steps.

You can use LaTeX to write your report. You can use [Overleaf](https://www.overleaf.com/) to write your report online. You can use [this template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn) to write your report.
Submit your report by sending an email to `moreno.laquatra@unikore.it`. It should include a link to your GitHub repository and the PDF of your report (3 pages max). -->

```{admonition} Final project - 1 week
As a final assignment for the course, you are asked to provide a short report presenting an idea on **how and where** you would use the tools presented in this course in your research. You can provide a brief description of the data you would use, the model you would use and the results you would expect to obtain. 
Even if not directly related to image or audio processing, you are free to choose the model you want to use (CNN, transformer, etc.). If you wish, you can provide a draft implementation of your idea. You can use the code from the previous exercises as a starting point.

Use LaTeX to write your report. You can use [Overleaf](https://www.overleaf.com/) to write your report online. You can use [this template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn) to write your report. Submit your report by sending an email to [moreno.laquatra@unikore.it](mailto:moreno.laquatra@unikore.it).
```