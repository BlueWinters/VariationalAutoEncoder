
import os as os
import numpy as np
import shutil as sh
import matplotlib.pyplot as plt
import scipy.misc
from datetime import datetime



def average_loss(ave_lost_list, step_loss_list, div):
    assert len(ave_lost_list) == len(step_loss_list)
    for n in range(len(ave_lost_list)):
        ave_lost_list[n] += step_loss_list[n] / div


def save_grid_images(images, save_path_and_name, nx=20, ny=20, size=32, chl=1):
    plt.cla()
    if chl == 3:
        stack_images = np.zeros([ny*size, nx*size, chl])
        for j in range(ny):
            for i in range(nx):
                stack_images[j*size:(j+1)*size, i*size:(i+1)*size, :] = np.reshape(images[j*ny+i,:], [size,size,chl])
        scipy.misc.imsave(save_path_and_name, stack_images)
        # plt.imshow(stack_images)
        # plt.savefig(save_path_and_name)
    else:
        stack_images = np.zeros([ny*size, nx*size])
        for j in range(ny):
            for i in range(nx):
                stack_images[j*size:(j+1)*size, i*size:(i+1)*size] = np.reshape(images[j*ny+i,:], [size,size])
        scipy.misc.imsave(save_path_and_name, stack_images)
        # plt.xticks([]), plt.yticks([])
        # plt.imshow(stack_images, cmap='gray')
        # plt.savefig(save_path_and_name)

def save_scattered_image(z, labels, save_path_and_name, z_range, y_dim=10):
    def discrete_cmap(N, base_cmap=None):
        base = plt.cm.get_cmap(base_cmap)
        color_list = base(np.linspace(0, 1, N))
        cmap_name = base.name + str(N)
        return base.from_list(cmap_name, color_list, N)

    plt.scatter(z[:,0], z[:,1], c=np.argmax(labels, 1), marker='o', edgecolor='none', cmap=discrete_cmap(y_dim, 'jet'))
    axes = plt.gca()
    axes.set_xlim([-z_range, z_range])
    axes.set_ylim([-z_range, z_range])
    plt.savefig(save_path_and_name)

def get_mesh(z_range, nx=20, ny=20):
    z = np.rollaxis(np.mgrid[z_range:-z_range:nx * 1j, z_range:-z_range:ny * 1j], 0, 3)
    return np.reshape(z, [-1, 2])

def plot_loss(save_path, path_list):
    def read_txt(file):
        loss = []
        BCE = []
        KLD = []
        for liner in file:
            info = liner.split(',')
            loss.append(float(info[1].split(' ')[-1]))
            BCE.append(float(info[2].split(' ')[-1]))
            KLD.append(float(info[3].split(' ')[-1]))
        return loss, BCE, KLD

    color_list = plt.get_cmap('hsv', len(path_list)*2)
    for n, path in enumerate(path_list):
        file = open('../save/{}/train.txt'.format(path), 'r')
        print(path)
        loss, BCE, KLD = read_txt(file)
        plt.plot(loss, color=color_list(2*n), label=path)
        file.close()
    plt.legend(loc='upper right')
    plt.title('Sum Loss')
    plt.savefig('../save/Loss.png')
    plt.show()



if __name__ == '__main__':
    path_list = ['dim2', 'dim4', 'dim8', 'dim10', 'dim20', 'dim100']
    plot_loss('save', path_list)




