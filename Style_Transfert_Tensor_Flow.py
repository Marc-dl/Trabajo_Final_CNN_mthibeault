# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:57:53 2020

@author: Marc Thibeault
"""
from keras.preprocessing.image import load_img, save_img, img_to_array
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from IPython.display import Image, display
from pathlib import Path
from matplotlib import pyplot as plt

# Setup
folder = 'D:/Marc/Anaconda/Trabajo _final_Computer_Vision/content/'
base_image_path = f'{folder}Tuebingen_Neckarfront.jpg'
# style_image_path = f'{folder}La_noche_estrellada1.jpg'
style_image_path = f'{folder}composition-vii.jpg'
result_image_path = f'{folder}output/Adam/image_generated'
result_prefix = "image_generated"

# Weights of the different loss components
total_variation_weight = 0
style_weight = 1e-2
content_weight = 1e-4
iterations = 400

# Dimensions of the generated picture.
width, height = load_img(base_image_path).size
# img_nrows = height
img_nrows = 400
img_ncols = int(width * img_nrows / height)
noise_matrix = np.random.uniform(0,255,size=(height,width,3))
noise = np.clip(noise_matrix, 0, 255).astype('uint8')
save_img(f'{folder}white_noise_image.jpg',noise)
noise_image_path = f'{folder}white_noise_image.jpg'

# image preprocessing/deprocessing utilities
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)

def deprocess_image(x):
    x = x.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Calculo de las Loss
def gram_matrix(x):
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    Ml2 = int(style.shape[0] * style.shape[1])**2
    Nl2 = int(style.shape[2])**2
    return tf.reduce_sum(tf.square(S - C))/ (4.0 * Nl2 * Ml2)

def content_loss(base, combination):
    return tf.reduce_sum(tf.square(combination - base))

def total_variation_loss(x):
    a = tf.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, 1:, :img_ncols - 1, :])
    b = tf.square(
        x[:, :img_nrows - 1, :img_ncols - 1, :] - x[:, :img_nrows - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# base_image = preprocess_image(base_image_path)
# style_image = preprocess_image(style_image_path)
# combination_image = tf.Variable(preprocess_image(base_image_path))
# input_tensor = tf.concat([base_image, style_image, combination_image], axis=0)
model = vgg19.VGG19(weights='imagenet', include_top=False, pooling='avg')
print('Model loaded.')

outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

content_layer_name = "block4_conv2"
style_layer_names = ["block1_conv1","block2_conv1","block3_conv1","block4_conv1",
                     "block5_conv1"]

def compute_loss(combination_image, base_image, style_reference_image):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0)
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features)
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss

@tf.function
def compute_loss_and_grads(combination_image, base_image, style_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads

# Se define dos optimizadores: 
# Primero a simple gradient descent steps to minimize 
# the loss, and save the resulting image every 100 iterations. The learning 
# rate by 0.96 every 100 steps.

# optimizer = keras.optimizers.SGD(keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96))

# Y obviamente Adam que es el optimizador estrella de gradient descent. 
optimizer = tf.optimizers.Adam(learning_rate=0.2,beta_1=0.99,epsilon=1e-1)

base_image = preprocess_image(base_image_path)
style_image = preprocess_image(style_image_path)
combination_image = tf.Variable(preprocess_image(base_image_path))

loss_graficar = []
for i in range(1, iterations + 1):
    start_time = time.time()
    loss, grads = compute_loss_and_grads(
        combination_image, base_image, style_image)
    optimizer.apply_gradients([(grads, combination_image)])
    if i % 10 == 0:
        print("Iteration %d: loss=%.2f" % (i, loss))
        loss_graficar.append(loss)
        img = deprocess_image(combination_image.numpy())
        d = i
        fname = f'{result_image_path}_at_iteration_{d}.png'
        save_img(fname, img)
#       fname = result_prefix + "_at_iteration_%d.png" % i
        keras.preprocessing.image.save_img(fname, img)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


# fname = f'{result_image_path}output_at_iteration_{d}.png'
# save_img(fname, img)




display(Image(base_image_path))
display(Image(style_image_path))
display(Image(f'{result_image_path}_at_iteration_{d}.png'))

plt.plot(loss_graficar)
plt.xlabel('loss')
plt.ylabel('iterations')
plt.savefig(f'{result_image_path}loss_graficar.png')
plt.show() 


















