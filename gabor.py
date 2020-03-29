import numpy as np
import os
import sys

from scipy import ndimage as ndi
from skimage import io
from skimage.filters import gabor
from skimage.filters import gabor_kernel
from skimage import data, io


DATA_FOLDER = './BMP600/'

# 获取图片名称地址
def getFileNamesByPath(path='./BMP600'):  #传入存储的list
  """
    path: 图片目录
      _1.bmp为测试, _x.bmp为训练, 按照8：2的近似比例分配
    return 
  """
  test_files = []
  train_files = []
  for file in os.listdir(path):  
    file_path = os.path.join(path, file)  
    if '_1.bmp' in file_path:
      test_files.append(file)
    else:
      train_files.append(file);
  return train_files, test_files


def get_img_id(image_file):
  id = image_file.split('_')[0]
  return id


def compute_feats(image, kernels):
  feats = np.zeros((len(kernels), 2), dtype=np.double)
  for k, kernel in enumerate(kernels):
    filtered = ndi.convolve(image, kernel, mode='constant')
  return filtered


def gabor_filter(image):
  # prepare filter bank kernels
  kernels = []
  for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
      for frequency in (0.05, 0.1):
        kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
        kernels.append(kernel)

  return compute_feats(image, kernels)


# 获取最相近的图片id。
def get_matched_id(models, test_img):
  matched = models[0]
  min_distance = np.inf
  for i, model in enumerate(models):
    if (np.sum(np.bitwise_xor(test_img, models[i]['img'])) < min_distance):
      matched = models[i]
      min_distance = np.sum(np.bitwise_xor(test_img, models[i]['img']))

  return matched['img_id']


# ref https://scikit-image.org/docsgg/dev/api/skimage.filters.html#skimage.filters.gabor
def train(train_files):
  models = []
  for i, image_file in enumerate(train_files):
    image = io.imread(DATA_FOLDER + image_file)
    print('processing the file: {} ...'.format(image_file))
    models.append({
      'img_id': get_img_id(image_file),
      'img': gabor_filter(image)
    })
  return models


def test(test_files, models):
  count = 0
  for i, image_file in enumerate(test_files):
    image = io.imread(DATA_FOLDER + image_file)
    test_img = gabor_filter(image)
    matched_id = get_matched_id(models, test_img)
    test_id = get_img_id(image_file)
    if matched_id == test_id:
      count = count + 1
    print('识别比例： {}/{}'.format(count, len(test_files)))
  return


def process():
  train_files, test_files = getFileNamesByPath(DATA_FOLDER)
  models = train(train_files)
  test(test_files, models)


if __name__ == '__main__':
  process()
