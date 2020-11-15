import argparse
import torch
from tqdm import tqdm

import datasets
from torch.autograd import Variable
import torch.nn.functional as F

def test_outage(model, test_loader, num_devices, outages):
    model.eval()
    num_correct = 0
    for data, target in tqdm(test_loader, leave=False):
        for outage in outages:
            data[:, outage] = 0
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        predictions = model(data)
        cloud_pred = predictions[-1]
        loss = F.cross_entropy(cloud_pred, target, size_average=False).item()
        pred = cloud_pred.data.max(1, keepdim=True)[1]
        correct = (pred.view(-1) == target.view(-1)).long().sum().item()
        num_correct += correct

    N = len(test_loader.dataset)

    print("correct: {}/{}".format(num_correct, N))

    return 100. * (num_correct / N)

def trigger_device_and_cloud_models_separately(model, test_loader):
    # This function is an illustration of how to access the individual parts of the DDNN.
    # The model argument holds a trained DDNN instance, i.e. it is an instance of the DDNN class (defined in net.py)
    # and therefore we can directly access all of its attributes. So instead of executing the whole DDNN at once we can
    # for example access and execute its device and cloud layers separately which is demonstrated in the code parts
    # below.

    model.eval()

    # holds the number of correctly classified training samples
    num_total_correct = 0

    # Loop over the test data and invoke the device models as well as the cloud model for each batch of training
    # samples. The size of the batch depends on the value that is passed via the --batch-size argument in the __main__
    # part below.
    # In each iteration of the loop
    # * the batch_data holds a batch of test samples
    # * the batch_classification holds the actual classification of the current test batch
    for batch_data, batch_classification in tqdm(test_loader, leave=False):
        if torch.cuda.is_available():
            batch_data, batch_classification = batch_data.cuda(), batch_classification.cuda()

        # Loop over the devices in the DDNN model and invoke their corresponding layers.
        device_outputs = []
        for device_index in range(len(model.device_models)):
            # Get the model for the specific device with the current index.
            device_model = model.device_models[device_index]

            # Extract the input data for the device from the current batch.
            device_input = batch_data[:, device_index]

            # The "call"-operator invokes the forward() method of a toch.nn.Module object and also just returns the
            # output of the forward() method. So here it invokes the forward() method defined in DeviceModel (see
            # net.py) and therefore returns the following values:
            # * the device_output is the tensor output of the layers that are part of the device model - this is the
            #   data the we would send to the next layer if we don't take an early exit
            # * the device_prediction that would have to be used to decide if we want take the early exit (here we do
            #   not simulate the early exit logic so the device_prediction is not used)
            device_output, device_prediction = device_model(device_input)

            # We now could save the the intermediate output of the device models to disk via:
            # torch.save(device_output, "device_output.pth")
            # internally torch.save() just uses pickle (see https://docs.python.org/3/library/pickle.html) to serialize
            # PyTorch objects. Instead of writing the device_output tensor to disc we could also directly serialize it
            # with pickle and send it over the network to the next layer of the DDNN.

            # Instead of actually sending the device output over the network just append it to the device_outputs.
            device_outputs.append(device_output)

        # Concatenate the output tensors of the devices - this part corresponds to the "cloud aggregator" component
        # as described in the paper.
        cloud_input = torch.cat(device_outputs, dim=1)

        # Invoke the cloud model with the concatenated data from the devices and compute the final
        # classification of the DDNN.
        cloud_model_output = model.cloud_model(cloud_input)
        cloud_pool_output = model.pool(cloud_model_output)
        batch_size = batch_data.shape[0]
        normalized_output = cloud_pool_output.view(batch_size, -1)
        class_probabilities = model.classifier(normalized_output)

        # The final classification of the DDNN which is a tensor of shape (batch_size, 1). For each of the
        # samples in the current batch it holds the class label that has been predicted by the DDNN.
        predicted_classification = class_probabilities.max(1, keepdim=True)[1]

        # Compute the number of correctly classified samples in the current batch by comparing the predicted
        # classification to the actual classification.
        num_batch_correct = (predicted_classification.view(-1) == batch_classification.view(-1)).long().sum().item()
        num_total_correct += num_batch_correct

    # Print out how well the DDNN performed on the test set.
    num_total_samples = len(test_loader.dataset)
    print("Correctly classified {} of {} samples".format(num_total_correct, num_total_samples))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='DDNN Evaluation')
    parser.add_argument('--dataset-root', default='datasets/', help='dataset root folder')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', default='mnist', help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--model_path', default='models/model.pth',
                        help='output directory')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    data = datasets.get_dataset(args.dataset_root, args.dataset, args.batch_size, args.cuda)
    train_dataset, train_loader, test_dataset, test_loader = data
    x, _ = train_loader.__iter__().next()
    num_devices = x.shape[1]

    map_location = None
    if not torch.cuda.is_available():
        map_location = torch.device('cpu')
    model = torch.load(args.model_path, map_location=torch.device('cpu'))

    for i in range(num_devices):
        outages = [i]
        acc = test_outage(model, test_loader, num_devices, outages)
        print('Missing Device(s) {}: {:.4f}'.format(outages, acc))

    for i in range(1, num_devices + 1):
        outages = list(range(i, num_devices))
        acc = test_outage(model, test_loader, num_devices, outages)
        print('Missing Device(s) {}: {:.4f}'.format(outages, acc))

    trigger_device_and_cloud_models_separately(model, test_loader)
