from Dataset import *
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.restoration import denoise_nl_means
from skimage.morphology import closing, square
from sklearn.feature_extraction import image
from sklearn.neighbors import NearestNeighbors
from skimage.util import view_as_blocks, view_as_windows

import math
from scipy.signal import convolve2d
import cv2
import time

def local_intensity_diff(img, x, y):
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    num_neighbors = 0
    sum_intensity = 0
    for x_d, y_d in directions:
        if x+x_d < 0 or y+y_d < 0 or x+x_d >= img.shape[0] or y+y_d >= img.shape[1]:
            continue

        sum_intensity += img[x+x_d][y+y_d] - img[x][y]
        num_neighbors += 1

    return sum_intensity / num_neighbors

def detecting_crack_pixels(img, crack_pixel_percentage=0.15):
    '''
    id_map = np.zeros_like(img)
    id_per_intensity = dict()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            id_map[i][j] = local_intensity_diff(img, i, j)
            if img[i][j] not in id_per_intensity:
                id_per_intensity[img[i][j]] = id_map[i][j]
            else:
                id_per_intensity[img[i][j]] += id_map[i][j]


    T = max(id_per_intensity, key=id_per_intensity.get)
    plt.plot(id_per_intensity.values())
    plt.show()
    plt.clf()
    '''

    start_time = time.time()
    threshold = 128
    while True:
        crack_pixels = np.where(img <= threshold, 255, 0)
        if np.sum(crack_pixels)/255 < crack_pixels.shape[0]*crack_pixels.shape[1]*crack_pixel_percentage:
            break
        threshold -= 1

    print("crack pixels generation time elapsed: ", time.time() - start_time)

    return crack_pixels

def color_correction(img, img_smooth):
    standard_color = np.ones_like(img) * 128
    diff = standard_color - img_smooth
    corrected_img = img + diff
    corrected_img = np.where(corrected_img < 0, 0, corrected_img)
    corrected_img = np.where(corrected_img > 255, 255, corrected_img)
    return corrected_img # np.expand_dims(corrected_img, axis=2)


def mask_generation(img: Image):
    img = np.asarray(img.resize((509, 625)).convert('L'), dtype=np.uint8)
    # show_grayscale_image(img)

    img = np.where(img > 128, 128, img)

    img_smooth = gaussian_filter1d(img, sigma=100, axis=0)
    img_smooth = gaussian_filter1d(img_smooth, sigma=5, axis=1)

    corrected_img = color_correction(img, img_smooth)
    # show_grayscale_image(corrected_img)
    # corrected_img = cv2.equalizeHist(corrected_img)
    # show_grayscale_image(corrected_img)

    '''
    gabor_kernel = cv2.getGaborKernel(ksize=(10, 10), sigma=2, theta=30, lambd=16, gamma=1)
    img_gabor = convolve2d(corrected_img, gabor_kernel, mode='same', boundary='fill')
    show_grayscale_image(img_gabor)
    '''

    crack_pixels = detecting_crack_pixels(corrected_img, crack_pixel_percentage=0.04).astype('uint8')
    n_crack_pixels = np.sum(crack_pixels)/255
    # print(n_crack_pixels)
    # show_grayscale_image(crack_pixels)

    crack_prob_map = tensor_voting(
        crack_pixels=crack_pixels,
        radius=10,
        total_directions=4,
        sigma=10,
        tensor_size_threshold=5.5,
        tensor_shape_threshold=1.2
    ).astype('uint8')
    # show_grayscale_image(crack_prob_map)

    return corrected_img, crack_pixels, crack_prob_map


def tensor_voting(crack_pixels, radius, total_directions, sigma,
                  tensor_size_threshold, tensor_shape_threshold):
    indices = np.where(crack_pixels == 255)
    '''
    tensor_map = np.zeros((crack_pixels.shape[0], crack_pixels.shape[1], 4))
    for i, j in zip(indices[0], indices[1]):
        # print(i, j)
        patch = []
        radius = 20
        patch.append(i - radius if i - radius >= 0 else 0)
        patch.append(i + radius if i + radius < crack_pixels.shape[0] else crack_pixels.shape[0])
        patch.append(j - radius if j - radius >= 0 else 0)
        patch.append(j + radius if j + radius < crack_pixels.shape[1] else crack_pixels.shape[1])
        neighbor_indices = np.where(crack_pixels[patch[0]:patch[1], patch[2]:patch[3]])
        for i2, j2 in zip(neighbor_indices[0], neighbor_indices[1]):
            if i2 == i and j2 == j:
                continue
            if i2 != 1 and j2 != 2:
                continue
            # print(i, j, i2, j2)
            theta = math.atan2(i2-i, j2-j)
            l = ((i2-i)**2 + (j2-j)**2) ** (1/2)
            for rotate_angle in [0, np.pi/4, np.pi/2, np.pi*3/4]:
                cur_theta = theta+rotate_angle if theta+rotate_angle < np.pi else theta+rotate_angle-np.pi
                tensor = saliency_decay(cur_theta, l, 5).reshape((-1,))
                tensor_map[i2, j2] += tensor

    print(tensor_map[1, 2])
    print(crack_pixels/255)
    '''

    start_time = time.time()
    tensor_map = np.zeros((crack_pixels.shape[0], crack_pixels.shape[1], 4), dtype='float64')
    # [0, 1.0/8, 2.0/8, 3.0/8, 4.0/8, 5.0/8, 6.0/8, 7.0/8]
    # [0, np.pi/4, np.pi/2, np.pi*3/4]
    for frac in range(total_directions):
        rotate_angle = frac*1.0 / total_directions * np.pi
        k1, k2, k3, k4 = build_saliency_decay_kernel(radius, rotate_angle, sigma)
        tensor_map[:, :, 0] += convolve2d(crack_pixels / 255, k1, mode='same', boundary='fill')
        tensor_map[:, :, 1] += convolve2d(crack_pixels / 255, k2, mode='same', boundary='fill')
        tensor_map[:, :, 2] += convolve2d(crack_pixels / 255, k3, mode='same', boundary='fill')
        tensor_map[:, :, 3] += convolve2d(crack_pixels / 255, k4, mode='same', boundary='fill')

    print("tensor voting time elapsed: ", time.time()-start_time)

    tensor_map = tensor_map.reshape(crack_pixels.shape[0], crack_pixels.shape[1], 2, 2)
    eigs = np.linalg.eig(tensor_map)[0]
    eigs1, eigs2 = eigs[:, :, 0], eigs[:, :, 1]
    eigs_max, eigs_min = np.maximum(eigs1, eigs2), np.minimum(eigs1, eigs2)
    eigs_min = np.where(eigs_min < 0.0005, 0.0005, eigs_min)

    crack_pixels = np.where((abs(eigs1 - eigs2) > tensor_size_threshold) &
                            (eigs_max/eigs_min > tensor_shape_threshold) & (crack_pixels), 255, 0)
    return crack_pixels


def build_saliency_decay_kernel(radius, rotation, sigma):
    kernel = np.zeros((radius*2+1, radius*2+1, 4))
    for i in range(0, radius*2+1):
        for j in range(0, radius*2+1):
            if i == radius and j == radius:
                continue
            theta = math.atan2(i - radius, j - radius)
            theta = theta + rotation if theta + rotation < np.pi else theta + rotation - np.pi
            l = ((i - radius) ** 2 + (j - radius) ** 2) ** (1 / 2)
            tensor = saliency_decay(theta, l, sigma).reshape((-1,))
            kernel[i, j] += tensor

    return kernel[:, :, 0], kernel[:, :, 1], kernel[:, :, 2], kernel[:, :, 3]


def saliency_decay(theta, l, sigma) -> np.array:
    s = l / np.sinc(theta)
    k = 2*np.sin(theta) / l
    c = -16*np.log(0.1)*(sigma-1) / np.pi
    return np.exp(-(s**2 + c*(k**2)) / (sigma**2)) * np.array([[-np.sin(2*theta)], [np.cos(2*theta)]]) *\
           np.array([[-np.sin(2*theta), np.cos(2*theta)]])


def patch_similarity(img, patch_size, percentage_taken):
    patch_step_size = 4
    block_step_size = int(patch_size[1]/1.5)

    patches = view_as_windows(img, patch_size, step=patch_step_size).reshape(-1, patch_size[0]*patch_size[1])

    blocks = view_as_windows(img, patch_size, step=block_step_size)
    total_blocks, num_patches_in_a_row = blocks.shape[0]*blocks.shape[1], blocks.shape[1]
    blocks = blocks.reshape(-1, patch_size[0]*patch_size[1])
    show_grayscale_image(blocks[0].reshape(patch_size))

    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=40, algorithm='ball_tree').fit(patches)
    print("NN initialized, time elapsed: %f"%(time.time()-start_time))
    distances, indices = nbrs.kneighbors(blocks)
    print("NN found, time elapsed: %f"%(time.time()-start_time))
    total_dist = np.sum(distances, axis=1)

    k = int(total_blocks*percentage_taken)
    print(k)
    max_indices = np.argpartition(total_dist, -k)[-k:]
    return [((i//num_patches_in_a_row)*block_step_size, (i%num_patches_in_a_row)*block_step_size) for i in max_indices]


def show_patches(img, patches_indices, patch_size, background_color=0):
    img_out = np.ones_like(img)*background_color
    for (i, j) in patches_indices:
        img_out[i:i+patch_size[0], j:j+patch_size[1]] = img[i:i+patch_size[0], j:j+patch_size[1]]
        # show_grayscale_image(img_out)

    return img_out


def show_rgb_image(img):
    plt.imshow(img, interpolation='nearest')
    plt.show()
    plt.clf()

def show_grayscale_image(img):
    plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    plt.show()
    plt.clf()

if __name__ == '__main__':
    dataset = CrackClassification(
        data_dir=str(Path('D:\\pickle_files')),
        transform=None,
        mode='crack_type_group',
        augmented_data_dir=None,
    )

    testing = False
    if testing:

        data = dataset[12] #30000? 80000, 90000
        # test: 12, 40000
        print(data['sev'])
        print(data['y'])
        show_rgb_image(data['img'])

        img = np.asarray(data['img'].resize((509, 625)).convert('L'), dtype=np.uint8)
        img = np.where(img > 128, 128, img)
        show_grayscale_image(img)

        img_smooth = gaussian_filter1d(img, sigma=100, axis=0)
        # show_grayscale_image(img_smooth)
        img_smooth = gaussian_filter1d(img_smooth, sigma=10, axis=1)
        # img_smooth = closing(img_smooth, square(10))
        # show_grayscale_image(img_smooth)
        # img_smooth = gaussian_filter(img_smooth, sigma=5)
        # show_grayscale_image(img_smooth)

        corrected_img = color_correction(img, img_smooth)
        # corrected_img = gaussian_filter(corrected_img, sigma=2)
        show_grayscale_image(corrected_img)

        crack_pixels = detecting_crack_pixels(corrected_img)
        crack_pixels_img = Image.fromarray(crack_pixels)
        show_grayscale_image(crack_pixels_img)

        img_blur = gaussian_filter(img, sigma=3)
        show_grayscale_image(img_blur)

        '''
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection
    
        show_grayscale_image(sobelx)
        '''

        '''
        patch_size = (50, 32)
        percentage_taken = 0.5
        most_likely_patches = patch_similarity(crack_pixels, patch_size, percentage_taken)
        img_out = show_patches(crack_pixels, most_likely_patches, patch_size, background_color=0)
        show_grayscale_image(img_out)
    
        crack_pixels = detecting_crack_pixels(img_out, crack_pixel_percentage=0.10)
        crack_pixels_img = Image.fromarray(crack_pixels)
        show_grayscale_image(crack_pixels_img)
        '''

        crack_pixels = tensor_voting(crack_pixels)
        # crack_pixels = tensor_voting(crack_pixels)

    if not testing:
        for i, data in enumerate(dataset):
            print(i)
            if i < 176361:
                continue
            corrected_img, img_mask_raw, img_mask_refined = mask_generation(data['img'])
            Image.fromarray(corrected_img).save(str(Path('D:\\image_augmentations\\%07d_corrected.png' % i)))
            Image.fromarray(img_mask_raw).save(str(Path('D:\\image_augmentations\\%07d_mask_raw.png' %i )))
            Image.fromarray(img_mask_refined).save(str(Path('D:\\image_augmentations\\%07d_mask_refined.png' % i)))

