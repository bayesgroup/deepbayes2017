import sys
import os

import numpy as np

import matplotlib.pylab as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def load_dataset():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, filename)

    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    X = {}
    y = {}
    
    X_train= load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    return X_train, y_train, X_val, y_val, X_test, y_test

def iterate_minibatches(inputs, targets=None, batchsize=20, present=None, shuffle=True):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        if targets is None:
            yield inputs[excerpt]
        elif present is None:
            yield inputs[excerpt], targets[excerpt]
        else:
            yield inputs[excerpt], targets[excerpt], present[excerpt]
            
def plot_reconstructions(x_test, reconstruction_func):
    decoded_imgs = reconstruction_func(x_test)

    indices = np.random.choice(x_test.shape[0], 64)
    
    n = x_test.shape[0]  # how many digits we will display
    
    fig, axes = plt.subplots(8, 16, figsize=(16, 8),
        subplot_kw={'xticks': [], 'yticks': []}
    )
    fig.subplots_adjust(hspace=0.04, wspace=0.02)

    for ax, i in zip(axes[:, :8].flat, indices):
        ax.imshow(x_test[i].reshape((28, 28)), cmap='gray')
        
    for ax, i in zip(axes[:, 8:].flat, indices):
        ax.imshow(decoded_imgs[i].reshape((28, 28)), cmap='gray')
        
    plt.show()
    

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    #im = OffsetImage(image.reshape((-1, 28, 28)), zoom=zoom)
    #x, y = np.atleast_1d(x, y)
    artists = []
    #assert len(x) == len(y) == len(image)
    n = len(x)
    for i in range(n):
        im = OffsetImage(image[i], zoom=zoom, cmap='gray')
        ab = AnnotationBbox(im, (x[i], y[i]), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def plot_hidden_space(x_test, encode_func, zoom=0.5):
    encoded = encode_func(x_test)
    
    fig, ax = plt.subplots(figsize=(11, 11))
    imscatter(encoded[:, 0], encoded[:, 1], x_test.reshape((-1, 28, 28)), zoom=zoom, ax=ax)
    
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    plt.gray()
    plt.show()