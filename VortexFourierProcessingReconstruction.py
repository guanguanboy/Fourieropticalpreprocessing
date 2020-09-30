import tensorflow as  tf
import numpy as np
# Load datasets - clothes and numbers
# Since only one is used in training you can comment one of them
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
(x_num, y_num), (x_test_num, y_test_num) = tf.keras.datasets.mnist.load_data()

print(type(x_num))
print(type(y_test_num))
print(x_num.shape)
print(y_test_num.shape)


import matplotlib.pyplot as plt
from keras import optimizers as kopt
from keras.models import Sequential
from keras.layers import Dense
import os

# Geometry definition
N=28
x = np.linspace(-5, 5-10/N, N)
[X,Y] = np.meshgrid(x,x)
thet = np.arctan2(Y,X)

# Normalize datasets
x_train = x_train.astype('float32')/255
x_num = x_num.astype('float32')/255
x_test = x_test.astype('float32')/255
x_test_num = x_test_num.astype('float32')/255

# Allocate the memory for vortex transformed intensity patterns
A = np.zeros([x_train.shape[0],28,28])
B = np.zeros([x_train.shape[0],28,28])
C = np.zeros([x_train.shape[0],28,28])

# Wide Gaussian, you can change this parameter and see what happens
rad =22
G = np.exp( -(X**2 + Y**2)/rad)
# Top charges
m1=1
m2=3
m3=5
#Vortices themselves
V1=np.exp(1j*m1*thet)
V2=np.exp(1j*m2*thet)
V3=np.exp(1j*m3*thet)
# Gaussian modulated vortices
VG1 = V1
VG2 = V2
VG3 = V3
# Iamginary Pi
impi = 1j*np.pi
# The cycle that computes the intensity patterns
# Computationally expensive, I would recommend saving the A, B and C after they are computed for the first time
# The 100000 factor is used to normalize the inputs to CNN
# Create the training data
scale = 1000
for k in range(0,x_train.shape[0]):
    TF1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG1*np.exp(impi*x_train[k]))))
    TF2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG2*np.exp(impi*x_train[k]))))
    TF3 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG3*np.exp(impi*x_train[k]))))
    A[k]=((np.abs(TF1))**2)/scale
    B[k]=((np.abs(TF2))**2)/scale
    C[k]=((np.abs(TF3))**2)/scale



# Allocate the memory for vortex transformed intensity patterns (test data)
At = np.zeros([x_test.shape[0],28,28])
Bt = np.zeros([x_test.shape[0],28,28])
Ct = np.zeros([x_test.shape[0],28,28])

# Create the test data
scale = 1000
for k in range(0,x_test.shape[0]):
    TF1 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG1*np.exp(impi*x_test[k]))))
    TF2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG2*np.exp(impi*x_test[k]))))
    TF3 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(VG3*np.exp(impi*x_test[k]))))
    At[k]=((np.abs(TF1))**2)/scale
    Bt[k]=((np.abs(TF2))**2)/scale
    Ct[k]=((np.abs(TF3))**2)/scale


# Flatten the training set
xr = x_train.reshape(x_train.shape[0], 28**2)

# Flatten the arrays and stack two of them together

A = A.reshape(x_train.shape[0], 28 ** 2)
B = B.reshape(x_train.shape[0], 28 ** 2)

Cd = np.hstack((A, B))
print(Cd.shape)

# Flatten the arrays and stack two of them together


At = At.reshape(x_test.shape[0], 28 ** 2)
Bt = Bt.reshape(x_test.shape[0], 28 ** 2)

Cdt = np.hstack((At, Bt))
Cdt_show = Cdt.reshape(x_test.shape[0], 28, 56)



print('Cdt.shape: ')
print(Cdt.shape)

print('Cdt.type: ')
print(type(Cdt))

print('x_test_type:')
print(type(x_test))

print(x_test.shape)

"""
#读取图片作为输入训练网络
image_root_dir = 'D:/DataSets/Result/train'
#image_path = os.path.join(image_root_dir, self.type_name)
img_name_list = os.listdir(image_root_dir)  # 得到path目录下所有图片名称的一个list

from skimage import io

image_list = []

image_count = len(img_name_list)
print('image_count:')
print(image_count)
for i in range(image_count):
    img_item_path = os.path.join(image_root_dir, img_name_list[i])
    img=io.imread(img_item_path)
    image_list.append(img)

#保存Cdt_show
image_save_root_dir = 'D:/DataSets/Result/saved'

for i in range(25):
    img_item_path = os.path.join(image_save_root_dir, img_name_list[i])
    io.imsave(img_item_path, Cdt_show[i])

print('image_array_type')
print(type(image_list))

image_array = np.array(image_list)
print(image_array.shape)
print(type(image_list))

CdP = image_array.reshape(x_train.shape[0], 2*28**2)
print('CdP shape')
print(CdP.shape)
"""

# Create the model with two layers

# The performance depends on the version of Tensorflow
# The smallest MSE does not correspond to the smallest SSIM (True for the linear model). You might want to stop training earlier
ep = 5
# Linear-Non-linear model is the simplest, but might require additional training
# Fully-linear needs threholding and tinkering with number of epochs, but achieves higher SSIM
# The greedy models with 4 or 5/6 vortices need more hidden units, more training, maybe dropout layer. A little bit of a
# headache.
# Number of nodes in hidden layer
Nh = 1400
model = Sequential()
model.add(Dense(Nh, input_dim=2*28**2, activation = 'linear', use_bias=False))
#model.add(Dropout(0.06))
model.add(Dense(28**2, input_dim=Nh, activation = 'relu', use_bias=True))
# For fully linear model we use threshold to get rid of "background"
model.compile(loss='mean_squared_error', optimizer='adam')
# Training section
St_tr = 30
En_tr = 50000
story = model.fit(CdP[St_tr:En_tr], xr[St_tr:En_tr], epochs=ep)


model.summary()

# Use model to predict the shapes
St_te = 0
En_te = 10000
x_pr = model.predict(CdP[St_te:En_te])
x_pred = x_pr.reshape(x_pr.shape[0], 28, 28)


plt.figure('predicted', figsize = (10,10))
#vlist= [4, 8, 9, 3, 16, 22, 15, 14, 19]
z=1
for i in range(0,25):
    plt.subplot(5,5,z)
    plt.imshow(x_pred[i], cmap='inferno')
    # Threshold used for linear model
    #plt.imshow(x_pred[i]*(x_pred[i]>0.2), cmap='inferno')
    plt.axis('off')
    z+=1

plt.show()

plt.figure('cdT', figsize = (10,10))

#vlist= [4, 8, 9, 3, 16, 22, 15, 14, 19]
z=1
for i in range(0,25):
    plt.subplot(5,5,z)
    plt.imshow(x_test[i], cmap='inferno')
    # Threshold used for linear model
    #plt.imshow(x_pred[i]*(x_pred[i]>0.2), cmap='inferno')
    plt.axis('off')
    z+=1

plt.show()

plt.figure('original', figsize = (10,10))
#vlist= [4, 8, 9, 3, 16, 22, 15, 14, 19]
z=1
for i in range(0,25):
    plt.subplot(5,5,z)
    plt.imshow(Cdt_show[i], cmap='inferno')
    # Threshold used for linear model
    #plt.imshow(x_pred[i]*(x_pred[i]>0.2), cmap='inferno')
    plt.axis('off')
    z+=1

plt.show()

from skimage.metrics import structural_similarity as ssim

dss = np.zeros((x_pred.shape[0], 1))
# Create a comparison set, from a test set
cmp_set = x_test[St_te:En_te]
for k in range(0, (En_te - St_te)):
    # For linear model we need a thresholding operation, for linear-non-linear it's not necessary
    # x_pred[k]*(x_pred[k]>0.18)
    dss[k] = ssim(x_pred[k], cmp_set[k])

SSIM = np.mean(dss)
print(SSIM)