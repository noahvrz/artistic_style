import argparse
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision import datasets, models, transforms
from torchvision.models.squeezenet import SqueezeNet, model_urls

INPUT_SIZE = 224

parser = argparse.ArgumentParser(description='Image Recognition using Neural Networks')
parser.add_argument('--batch-size', type=int, default=20,
					help='batch size for training (default: 20)')
parser.add_argument('--folder', default='/mnt/research/gis/users/nverzani/ML/art_images',
					help='folder where images are stored')
parser.add_argument('--num-classes', type=int, default=25,
					help='number of target classes (default: 25)')
parser.add_argument('--epochs', type=int, default=30,
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--nocuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from latest checkpoint')
parser.add_argument('--print-interval', type=int, default=100,
					help='interval at which to print training loss')
parser.add_argument('--validate', action='store_true', default=False,
                    help='test model on validation set')

args = parser.parse_args()

# Check for CUDA availability
args.cuda = not args.nocuda and torch.cuda.is_available()
if args.cuda:
	print("Using CUDA device {}".format(torch.cuda.current_device()))
if not args.nocuda and not torch.cuda.is_available():
	print("Not using CUDA because it isn't available.")

# Create datasets and dataloaders to load them

# When testing, do some slight random scales and crops.
# Normalize in the same way as the pre-trained network.
train_transforms = transforms.Compose([
	transforms.RandomSizedCrop(INPUT_SIZE),
	transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# When testing and validating, just use center crop
test_transforms = transforms.Compose([
	transforms.Scale(256),
	transforms.CenterCrop(INPUT_SIZE),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Additional arguments for CUDA support
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# The images are located in args.folder/{train/val/test}/{style}/*.jpg
train_dataset = datasets.ImageFolder(os.path.join(args.folder, 'train'), train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

val_dataset = datasets.ImageFolder(os.path.join(args.folder, 'val'), test_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

# Create a V1.1 Squeezenet
model = SqueezeNet(version=1.1, num_classes=args.num_classes)

# Download a pretrained model, replace the last layer with a newly initialized one of correct size
state_dict = torch.utils.model_zoo.load_url(model_urls['squeezenet1_1'])
state_dict['classifier.1.weight'] = model.state_dict()['classifier.1.weight']
state_dict['classifier.1.bias'] = model.state_dict()['classifier.1.bias']

# Load the downloaded parameters into the squeeze net
model.load_state_dict(state_dict)

if args.cuda:
	model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

loss_function = nn.CrossEntropyLoss()

if args.cuda:
	loss_function = loss_function.cuda()

def train(epoch):
	model.train()

	for i, (image, style) in enumerate(train_loader):
		if args.cuda:
			image = image.cuda(async=True)
			style = style.cuda(async=True)

		image, style = Variable(image), Variable(style)

		optimizer.zero_grad()
		guess = model(image)
		loss = loss_function(guess, style)
		loss.backward()
		optimizer.step()

		if i % args.print_interval == 0:
			print("[Epoch {}][Batch {}/{}]	Training loss: {}".format(
				epoch, i, len(train_loader), loss.data[0]))

def val(epoch):
	model.eval()

	loss = 0
	num_correct = 0
	for i, (image, style) in enumerate(val_loader):

		if args.cuda:
			image = image.cuda(async=True)
			style = style.cuda(async=True)

		image, style = Variable(image, volatile=True), Variable(style, volatile=True)

		guess = model(image)
		loss += loss_function(guess, style).data[0]

		prediction = guess.data.max(1)[1]
		num_correct += prediction.eq(style.data).cpu().sum()

	loss /= len(val_loader)
	accuracy = num_correct / len(val_dataset)

	print("[Epoch {}]	Average Testing Loss: {}, Accuracy: {:.2}%".format(
		epoch, loss, 100*accuracy))

	return accuracy

best_accuracy = 0
start_epoch = 1

# If we want to resume from the last checkpoint, load its information
if args.resume:
	save = torch.load('checkpoint.pth')
	start_epoch = save['epoch']
	best_accuracy = save['best_accuracy']

	model.load_state_dict(save['model'])
	optimizer.load_state_dict(save['optimizer'])

if args.validate:
	val(0)

# Train and validate epochs until we reach the maximum
for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    accuracy = val(epoch)

    # Save a checkpoint after every epoch
    save = {
    	'epoch': epoch,
    	'best_accuracy': best_accuracy,
    	'model': model.state_dict(),
    	'optimizer': optimizer.state_dict()
    }
    torch.save(save, 'checkpoint.pth')

    # Save it seperately if it has the best accuracy
    if accuracy > best_accuracy:
    	torch.save(save, 'best.pth')
