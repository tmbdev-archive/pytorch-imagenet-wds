**This repository illustrates how to minimally change an existing PyTorch program
to use WebDataset. If you are starting from scratch, you probably want to write your code
differently. In particular, for large multinode training, your code becomes simpler if
you don't use epochs and instead train on a continuous stream of batches.**

**Have a look at https://github.com/tmbdev/wds-distributed to understand how
sharding works with DistributedDataParallel training.**

**NB: This repository needs to be updated to the latest version of WebDataset.**

# Introduction

WebDataset is an implementation of the IterableDataset API in PyTorch.
WebDataset can handle training sets that are petabytes large, provides
efficient I/O when loading data over the network or from cheap hard
disks, permits efficient training with datasets stored on web servers
(hence the name), speeds up the startup and debugging of deep learning
jobs. WebDataset is being integrated into the core PyTorch.

This repo contains a simple example of using
[WebDataset](http://www.github.com/tmbdev/webdataset) for training
models on ImageNet using PyTorch.  The code is a derivative of
the PyTorch Imagenet training example from the [PyTorch examples
folder](https://github.com/pytorch/examples/tree/master/imagenet).

The purpose of this repository is just to demonstrate how WebDataset
can be integrated into an existing, well-known deep learning job.
Since ImageNet is a fairly small dataset, you may only see a small
difference in load speed when training from local SSD, though you may
still benefit from faster startup times.

The original code is in `main-orig.py` and the WebDataset-based code is in
`main-wds.py`. The WebDataset-based code leaves most of the original code
untouched and only changes the Dataset/DataLoader portions of the code.

# Generating the Shards

Before training with WebDataset, you need a sharded dataset. Datasets are becoming
available directly in WebDataset format, but since Imagenet is not freely
redistributable, you have to generate the WebDataset version of Imagenet yourself
from your Imagenet distribution. The script `makeshards.py` will do this for you.

Let's say you have installed ImageNet in `/data/imagenet` and you can
train with `torchvision.datasets.ImageNet("/data/imagenet")`. To transform
that data into shards, you can use:

```Bash
$ ln -s /data/imagenet ./data
$ mkdir ./shards
$ ./run venv         # create the virtual environment
$ ./run makeshards   # create the shards
```

This should take a few minutes, and eventually, you should end up with 1282
training shards and 50 validation shards. You can split your dataset into larger
or smaller shards, depending on your needs. These shards happen to contain
1000 training exaamples each.

Have a look at `makeshards.py` to see how the shards are written and note
how the script piggy-backs on the original PyTorch `ImageNet` class for
reading the metadata.

# Running Training

You can simply run training using the original and the new data set implementations
using

```Bash
$ `./run train`.
```

Have a look at `main-wds.py` to see the different options for selecting different
dataset implementations, different augmentation methods

# Changes From Original

The changes from the original are:

- creation of loaders has been refactored into separate functions
  (`make_train_loader_orig` and `make_train_loader_wds`)
- the original code called `dataset.sampler.set_epoch`, but that method
  is not available on iterable datasets, so the call is conditional
- for the original data loader, the directory is now an option (rather
  than a positional argument)
- a few new command line options

You can see the differences by typing:

```Bash
$ diff main-orig.py main-wds.py
```

or

```Bash
$ meld main-orig.py main-wds.py
```
