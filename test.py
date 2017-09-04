import torch
import torch.multiprocessing as mp

print(torch.cuda.is_available())

def worker(rank):
	print(rank)

if __name__ == '__main__':

	# args = parser.parse_args()
	
	# #os.environ['OMP_NUM_THREADS'] = '1'

	# torch.manual_seed(args.seed)

	# # Create a V1.1 Squeezenet, share its memory
	# model = SqueezeNet(version=1.1, num_classes=NUM_CLASSES)
	# model.share_memory()

	# worker_args = (args, model)
	
	# if args.resume:
	# 	save = torch.load('checkpoint.pth')
	# 	start_epoch = save['epoch']
	# 	best_accuracy = save['best_accuracy']

	# 	model.load_state_dict(save['model'])
	# 	optimizer.load_state_dict(save['optimizer'])

	# 	worker_args += (start_epoch, best_accuracy,)
	# else:
	# 	# Download a pretrained model, replace the last layer with a newly initialized one
	# 	state_dict = torch.utils.model_zoo.load_url(model_urls['squeezenet1_1'])
	# 	state_dict['classifier.1.weight'] = model.state_dict()['classifier.1.weight']
	# 	state_dict['classifier.1.bias'] = model.state_dict()['classifier.1.bias']

	# 	# Load the new parameters into the squeeze net
	# 	model.load_state_dict(state_dict)

	# Start up the multiple processes
	processes = []
	for r in range(args.num_processes):
		p = mp.Process(target=worker, args=(r,)+worker_args)
		p.start()
		processes.append(p)
	for p in processes:
		p.join()