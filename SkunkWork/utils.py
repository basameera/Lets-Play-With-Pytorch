"""Python Tools by Bassandaruwan"""
import os
from PIL import Image
import datetime

def clog(*args):
    msg = '>>> '+str(datetime.datetime.now()) + ' :'
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


def getListOfFilesX(dirName, ext=['.jpg', '.png']):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over a4ll the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFilesX(fullPath)
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


def imgResize(fileList, size=200, save_folder='save'):
    size = 256, 256
    for id, path in enumerate(fileList):
        print(id, path)
        im = Image.open(path)
        im = im.resize(size, Image.ANTIALIAS)
        im.save(save_folder+'/'+str(id)+'.jpg')


def imgResize(source_folder='data', destination_folder='save', size=256, keep_aspect_ratio=True):
    fileList = getListOfFilesX(source_folder)

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

    # prettyPrint(dict(config))
    # prettyPrint(dict(config), 'config')

    filesCS = getListOfFilesX('data/old/CS')
    filesMD = getListOfFilesX('data/old/MD')

    print('cs data len', len(filesCS))
    print('md data len', len(filesMD))

    print('cs data size', getDatasetSizeOnDisk(filesCS))
    print('md data size', getDatasetSizeOnDisk(filesMD))

    # imgResize(source_folder='data/old/MD', destination_folder='data/new/MD')

    clog('someting')
    clog('sdfsdf', 'sdfsdfsdlfjsdfj')


# run
if __name__ == '__main__':
    main()
