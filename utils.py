import numpy as np
import os, struct
from array import array
import matplotlib
import matplotlib.pyplot as plt

kwargs = {'linewidth' : 3.5}
font = {'weight' : 'normal', 'size'   : 24}
matplotlib.rc('font', **font)

##################
### mnist dataloader
##################
def read_idx(filename):
    """Read files"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

def normalize_stats_image_by_image(images):
    """Normalizing images"""
    mean = images.mean(axis=(1,2), keepdims=True)
    stdev = images.std(axis=(1,2), keepdims=True)
    return (images - mean) / stdev    
    
def mnist_data_loader(img_dir, labels_dir):
    """Loading and normalizing images"""
    # train images
    train_mnist_imgs = read_idx(img_dir)
    train_mnist_labels = read_idx(labels_dir)
    train_mnist_imgs = train_mnist_imgs.astype(np.float32)
    train_mnist_labels = train_mnist_labels.astype(np.int)
    
    # pick 6/8 images
    train_mask_6_8 = (train_mnist_labels == 6) | (train_mnist_labels == 8)
    train_images_6_8 = train_mnist_imgs[train_mask_6_8]
    train_labels_6_8 = (train_mnist_labels[train_mask_6_8] == 8).astype(np.int)
    train_labels_6_8[train_labels_6_8==0]=-1
    
    #show loaded images and labels
    #example_images = np.concatenate(train_images_6_8[:10], axis=1)
    #example_labels = train_labels_6_8[:10]
    #plt.imshow(example_images)
    #plt.grid(False)
    #plt.show()

    # normalize images
    train_images_6_8 = normalize_stats_image_by_image(train_images_6_8)
    return train_images_6_8, train_labels_6_8


##################
### plot functions
##################
#def error_plot(ys, yscale='log'):
#    """plot errors"""
#    plt.figure(figsize=(8, 8))
#    plt.xlabel('Step')
#    plt.ylabel('Error')
#    plt.yscale(yscale)
#    plt.plot(range(len(ys)), ys, **kwargs)
    
#def acc_plot(ys):
#    """plot accuracies"""
#    plt.figure(figsize=(8, 8))
#    plt.xlabel('Step')
#    plt.ylabel('Acc')
#    plt.plot(range(len(ys)), ys, **kwargs)

def objective(w, _lambda, imgs, labels):
    """Calculate the objective function"""
    hinge_loss = 0
    for i in range(0,imgs.shape[0]):
        hinge_loss += np.max([0.0, np.float(1-labels[i]* (w.T @ imgs[i].T))])
#     print(hinge_loss, np.linalg.norm(w))
    losses = _lambda/2*np.linalg.norm(w)**2 + (1/imgs.shape[0])*hinge_loss
    return losses

def compute_accuracy(w, imgs, labels):
    """Compute the accuracy"""
    correct = 0
    for i in range(0, imgs.shape[0]):
        pred = w.T @ imgs[i].T>=0
        if pred==0:
            pred=-1
        correct += int(labels[i]==pred)
    acc = correct/imgs.shape[0]
    return acc


def proj(x, _lambda):
    """Projection of x onto an affine subspace --- 1/np.sqrt(_lambda) ball centered at the origin"""
    if np.linalg.norm(x)>(1/np.sqrt(_lambda)):
        x_proj = x/np.linalg.norm(x)*(1/np.sqrt(_lambda))
    else:
        x_proj = x
    return x_proj

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def get_cifar10(folder):
    tr_data = np.empty((0,32*32*3))
    tr_labels = np.empty(1)
    '''
    32x32x3
    '''
    for i in range(1,6):
        fname = folder + "data_batch_" + str(i)
        data_dict = unpickle(fname)
        # print(data_dict.keys())
        if i == 1:
            tr_data = data_dict[b'data']
            tr_labels = data_dict[b'labels']
        else:
            tr_data = np.vstack((tr_data, data_dict[b'data']))
            tr_labels = np.hstack((tr_labels, data_dict[b'labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    te_data = data_dict[b'data']
    te_labels = np.array(data_dict[b'labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    label_names = bm[b'label_names']

    mask1 = (tr_labels == 1) | (tr_labels == 7)
    tr_data1 = tr_data[mask1]
    tr_labels1 = (tr_labels[mask1] == 1).astype(np.int) * 2 - 1

    mask2 = (te_labels == 1) | (te_labels == 7)
    te_data1 = te_data[mask2]
    te_labels1 = (te_labels[mask2] == 1).astype(np.int) * 2 - 1
    print(tr_data1.shape)
    tr_data1 = (tr_data1 - tr_data1.mean(axis=1, keepdims=True)) / tr_data1.std(axis=1, keepdims=True)
    te_data1 = (te_data1 - te_data1.mean(axis=1, keepdims=True)) / te_data1.std(axis=1, keepdims=True)

    return tr_data1, tr_labels1, te_data1, te_labels1, label_names
