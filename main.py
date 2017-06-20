import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

from torchvision import datasets, models, transforms
from torchvision.models.squeezenet import SqueezeNet, model_urls

INPUT_SIZE = 224
NUM_CLASSES = 25

parser = argparse.ArgumentParser(description='Image Recognition using Neural Networks')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
					help='batch size for training (default: 20)')
parser.add_argument('--folder-name', default='/mnt/research/gis/users/nverzani/ML/art_images', metavar='F',
					help='folder where images are stored (default: images)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from latest checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print-interval', type=int, default=100, metavar=N,
					help='interval at which to print training loss')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
	torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# When testing, do some slight random scales and crops.
# Normalize in the same way as the pre-trained Squeezenet.
train_transforms = transforms.Compose([
	transforms.RandomSizedCrop(INPUT_SIZE),
	transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# When testing and validating, just use center crop
test_transforms = transforms.Compose([
	transforms.Scale(INPUT_SIZE),
	transforms.CenterCrop(256),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# The images are located in {train/val/test}/{style}/*.jpg
train_dataset = datasets.ImageFolder(args.folder_name+"/train", train_transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=true, **kwargs)

val_dataset = datasets.ImageFolder(args.folder_name+"/val", test_transforms)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=true, **kwargs)

test_dataset = datasets.ImageFolder(args.folder_name+"/test", test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=true, **kwargs)

# Create a V1.1 Squeezenet
model = SqueezeNet(version=1.1, num_classes=NUM_CLASSES)

# Download a pretrained model, replace the last layer with a newly initialized one
state_dict = torch.utils.model_zoo.load_url(model_urls['squeezenet1_1'])
state_dict['classifier.1.weight'] = model.state_dict()['classifier.1.weight']
state_dict['classifier.1.bias'] = model.state_dict()['classifier.1.bias']

# Load the new parameters into the squeeze net
model.load_state_dict(state_dict)

if args.cuda:
	model = model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

def train(epoch):
	model.train()

	for i, (image, style) in enumerate(train_loader):

		if args.cuda:
			image, style = image.cuda(), style.cuda()

		image, style = Variable(image), Variable(style)

		optimizer.zero_grad()
		guess = model(image)
		loss = nn.functional.CrossEntropyLoss(guess, style)
		loss.backward()
		optimizer.step()

		if i % args.print_interval == 0:
			print("[Epoch {}][Batch {}/{}]	Training loss: {}".format(
				epoch, i, len(train_loader), loss.data[0]))

def val(epoch):
	
	model.eval()

	loss = 0
	num_correct = 0
	for batch_num, (image, style) in enumerate(val_loader):

		if args.cuda:
			image, style = image.cuda(), style.cuda()

		image, style = Variable(image), Variable(style)

		guess = model(image)
		loss += nn.functional.CrossEntropyLoss(guess, style)
		
		prediction = guess.data.max(1)[1]
		num_correct += prediction.eq(style.data).cpu().sum()

	loss /= len(val_loader)
	accuracy = num_correct / len(val_dataset)

	print("[Epoch {}]	Average Testing Loss: {}, Accuracy: {}%".format(
		epoch, loss, 100*accuracy))

	return accuracy

best_accuracy = 0
start_epoch = 0

if args.resume:
	save = torch.load('checkpoint.pth')
	start_epoch = save['epoch']
	best_accuracy = save['best_accuracy']

	model.load_state_dict(save['model'])
	optimizer.load_state_dict(save['optimizer'])

for epoch in range(start_epoch, args.epochs + 1):
    train(epoch)
    accuracy = test(epoch)

    save = {
    	'epoch': epoch,
    	'best_accuracy': best_accuracy,
    	'model': model.state_dict(),
    	'optimizer': optimizer.state_dict()
    }

    torch.save(save, 'checkpoint.pth')

    if accuracy > best_accuracy:
    	torch.save(save, 'best.pth')
