"""SkunkWork Utils"""
import os
from PIL import Image
import datetime
import json

import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def clog(*args):
    msg = '>>> '+str(datetime.datetime.now()).split('.')[0] + ' :'
    for s in args:
        msg = msg + ' ' + str(s)
    print(msg)


def printLine(len, end='\n'):
    if isinstance(len, int):
        for _ in range(len):
            print('=', end='')
        print(end, end='')
    else:
        raise TypeError('Input should be an int value.')


def getMaxLen(input, output, prev_indent=0):
    if isinstance(input, dict):
        maxKeyLen = 0
        maxValLen = 0
        for key, value in input.items():
            if len(str(key)) > maxKeyLen:
                maxKeyLen = len(str(key))
            if len(str(value)) > maxValLen:
                maxValLen = len(str(value))

        rawValLen = 0
        # data
        for key, value in input.items():
            klen = len(str(key))

            if isinstance(value, dict):
                getMaxLen(value, output,
                          prev_indent=prev_indent + maxKeyLen + 1)
            else:
                if len(str(value)) > rawValLen:
                    rawValLen = len(str(value))

        a = maxKeyLen + prev_indent + rawValLen
        output.append(a)

    else:
        raise TypeError('Input should be a Dictionary object.')


def prettyPrint(input, heading='', prev_indent=0):

    if isinstance(input, dict):
        zzz = []
        getMaxLen(input, zzz)
        maxFooterLen = max(zzz)

        maxKeyLen = 0
        maxValLen = 0
        for key, value in input.items():
            if len(str(key)) > maxKeyLen:
                maxKeyLen = len(str(key))
            if len(str(value)) > maxValLen:
                maxValLen = len(str(value))

        a = maxFooterLen - len(str(heading))
        if heading == '':
            a += 2
        # header
        if prev_indent == 0:
            printLine(int(a/2), end='')
            if heading != '':
                print('', heading, '', end='')
            printLine(a - int(a/2))

        # data
        for key, value in input.items():
            for _ in range(prev_indent):
                print(' ', end='')
            print(str(key), end='')
            klen = len(str(key))
            for _ in range(maxKeyLen - klen):
                print(' ', end='')

            print(':', end='')

            if isinstance(value, dict):
                print('')
                prettyPrint(value, prev_indent=prev_indent + maxKeyLen + 1)
            else:
                print('', value)

        # footer
        if prev_indent == 0:
            printLine(maxFooterLen + 2)

    else:
        raise TypeError('Input should be a Dictionary object.')


def getListOfFiles(sourcePath, ext=['.jpg', '.png']):
    """getListOfFiles

    Arguments:
        sourcePath {[type]} -- [description]

    Keyword Arguments:
        ext {list} -- [description] (default: {['.jpg', '.png']})
    """
    listOfFile = os.listdir(sourcePath)
    allFiles = list()
    # Iterate over a4ll the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(sourcePath, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            if os.path.isfile(fullPath):
                _, f_ext = os.path.splitext(fullPath)
                if (f_ext.upper() in ext) or (f_ext.lower() in ext):
                    allFiles.append(fullPath)

    return allFiles


def getDatasetSizeOnDisk(fileList):
    size = 0
    ext = 'Bytes'
    for file in fileList:
        size += os.stat(file).st_size

    if size > 1e6:
        size /= 1e6
        ext = 'MB'
    if size > 1e3:
        size /= 1e3
        ext = 'kB'
    return size, ext


def getSplitByPercentage(train_percentage=0.8, len=0):
    if train_percentage > 0.0 and train_percentage < 1.0:
        train_p = int(train_percentage*len)
        valid_p = (len - train_p)//2
        return [train_p, valid_p, len - train_p - valid_p]
    else:
        raise ValueError('Value should be between 0 and 1.')


def imgResize(source_folder='data', destination_folder='save', size=256, keep_aspect_ratio=True):
    fileList = getListOfFiles(source_folder)

    for id, path in enumerate(fileList):
        print(id, ' | ', path, end=' | ')
        im = Image.open(path)
        width, height = im.size
        print((width, height), end=' -> ')

        new_size = (size, size)
        if keep_aspect_ratio:
            if height < width:
                new_size = ((size * width) // height, size)
            if height > width:
                new_size = (size, (size * height) // width)

        print(new_size)
        im = im.resize(new_size, Image.ANTIALIAS)
        im.save(destination_folder+'/'+str(id)+'.jpg')


def read_json(path='results/loss_data.json'):
    with open(path, 'r') as jfile:
        return json.loads(jfile.read())

# pytorch model summary
def model_summary(model, *input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]

            # only print nn modules
            if hasattr(nn, class_name):
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("\n----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) *
                           batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print(
        "Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------\n")
# main


def main():
    config = {
        "name": "Mnist_LeNet",
        "n_gpu": 1,

        "arch": {
            "type": "MnistModel",
            "args": {

            }
        },
        "data_loader": {
            "type": "MnistDataLoader",
            "args": {
                "data_dir": "data/",
                "batch_size": 64,
                "shuffle": True,
                "validation_split": 0.1,
                "num_workers": 2
            }
        },
        "optimizer": {
            "type": "Adam",
            "args": {
                "lr": 0.001,
                "weight_decay": 0,
                "amsgrad": True
            }
        },
        "loss": "nll_loss",
        "metrics": [
            "my_metric", "my_metric2"
        ],
        "lr_scheduler": {
            "type": "StepLR",
            "args": {
                "step_size": 50,
                "gamma": 0.1
            }
        },
        "trainer": {
            "epochs": 100,
            "save_dir": "saved/",
            "save_freq": 1,
            "verbosity": 2,
            "monitor": "min val_loss",
            "early_stop": 10,
            "tensorboardX": True,
        }
    }


# run
if __name__ == '__main__':
    main()
