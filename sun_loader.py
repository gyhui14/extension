import torch
import numpy as np
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from collections import defaultdict

def get_train_valid_loader(train_data_dir,
						   test_data_dir,
                           batch_size=64,
                           examples_per_label = None,
                           random_seed=1993,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=True):

	numclass = 397
	normalize = transforms.Normalize(
	    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	# load the training dataset
	train_dataset = torchvision.datasets.ImageFolder(train_data_dir,
	                 transforms.Compose([
	                     transforms.Scale(256),
	                     transforms.RandomSizedCrop(224),
	                     transforms.RandomHorizontalFlip(),
	                     transforms.ToTensor(),
	                     normalize,
	                 ]))

	if examples_per_label is not None:
		targets = np.asarray([s[1] for s in train_dataset.samples])
		num_train = len(train_dataset)
		indices = list(range(num_train))
		if shuffle:
		    np.random.seed(random_seed)
		    np.random.shuffle(indices)

		targets = targets[indices]
		train_idx_dic = defaultdict(list)
		for idx, item in enumerate(targets):
			if len(train_idx_dic[item]) != examples_per_label:
				train_idx_dic[item].append(idx)

		train_idx_balanced = []
		for key, item in train_idx_dic.items():
			train_idx_balanced += item

		train_sampler = SubsetRandomSampler(train_idx_balanced)
		
		train_loader = torch.utils.data.DataLoader(
		    train_dataset, batch_size=batch_size, sampler=train_sampler,
		    num_workers=num_workers, pin_memory=pin_memory,
		)
	else:
		train_loader = torch.utils.data.DataLoader(
		    train_dataset, batch_size=batch_size, shuffle=True,
		    num_workers=num_workers, pin_memory=pin_memory,
		)	

	# load the test dataset
	test_dataset = torchvision.datasets.ImageFolder(test_data_dir,
	                 transforms.Compose([
	                     transforms.Scale(256),
	                     transforms.CenterCrop(224),
	                     transforms.ToTensor(),
	                     normalize,
	                 ]))

	test_loader = torch.utils.data.DataLoader(
	    test_dataset, batch_size=batch_size, shuffle=False,
	    num_workers=num_workers, pin_memory=pin_memory,
	)

	return (train_loader, test_loader, numclass)