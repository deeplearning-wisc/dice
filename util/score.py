import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_msp_score(inputs, model, method_args):
    with torch.no_grad():
        outputs = model(inputs)
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores

def get_sofl_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)
    scores = -F.softmax(outputs, dim=1)[:, num_classes:].sum(dim=1).detach().cpu().numpy()

    return scores

def get_rowl_score(inputs, model, method_args, raw_score=False):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)

    if raw_score:
        scores = -1.0 * F.softmax(outputs, dim=1)[:, num_classes].float().detach().cpu().numpy()
    else:
        scores = -1.0 * (outputs.argmax(dim=1)==num_classes).float().detach().cpu().numpy()

    return scores

def get_atom_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    with torch.no_grad():
        outputs = model(inputs)
    #scores = -F.softmax(outputs, dim=1)[:, num_classes]
    scores = -1.0 * (F.softmax(outputs, dim=1)[:,-1]).float().detach().cpu().numpy()

    return scores

def get_odin_score(inputs, model, method_args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input

    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = Variable(inputs, requires_grad = True)
    outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_energy_score(inputs, model, method_args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input

    # temper = method_args['temperature']

    inputs = Variable(inputs, requires_grad = True)
    outputs = model(inputs)

    # Using temperature scaling
    # outputs = outputs / temper
    nnOutputs = outputs.data.cpu()
    scores = torch.logsumexp(nnOutputs, dim=1).numpy()

    return scores

def get_mahalanobis_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    sample_mean = method_args['sample_mean']
    precision = method_args['precision']
    magnitude = method_args['magnitude']
    regressor = method_args['regressor']
    num_output = method_args['num_output']
    Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude)
    scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]

    return scores

def get_score(inputs, model, method, method_args, raw_score=False):
    if method == "msp":
        scores = get_msp_score(inputs, model, method_args)
    elif method == "odin":
        scores = get_odin_score(inputs, model, method_args)
    elif method == "energy":
        scores = get_energy_score(inputs, model, method_args)
    elif method == "mahalanobis":
        scores = get_mahalanobis_score(inputs, model, method_args)
    elif method == "sofl":
        scores = get_sofl_score(inputs, model, method_args)
    elif method == "rowl":
        scores = get_rowl_score(inputs, model, method_args, raw_score)
    elif method == "atom":
        scores = get_atom_score(inputs, model, method_args)
    return scores