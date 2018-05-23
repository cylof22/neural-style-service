import os
from argparse import ArgumentParser
from keras import backend as K
import numpy as np
from vgg19model import VGG19Model
from keras.applications import vgg19
from keras.preprocessing.image import load_img, img_to_array

from scipy.optimize import fmin_l_bfgs_b
from PIL import Image
from sys import stderr

# default content, style weights, and the convergence error
# need to be added to the registered environment variables
content_weight = 0.025
style_weight = 1.0
total_variation_weight = 1e-4
source_paper = 'gatys'
iterations = 1

def getPreviewEnvs():
    content = os.environ["content"]
    style = os.environ["styles"]
    output = os.environ["output"]
    return (content, style, output)

def preprocess_image(image_path, height=None, width=None):
    height = 400 if not height else height
    width = width if width else int(width * height / height)
    img = load_img(image_path, target_size=(height, width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(x):
    # Remove zero-center by mean pixel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def content_loss(base, combination):
    return K.sum(K.square(combination - base))

def style_loss(style, combination, height, width):
    
    def build_gram_matrix(x):
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
        gram_matrix = K.dot(features, K.transpose(features))
        return gram_matrix

    S = build_gram_matrix(style)
    C = build_gram_matrix(combination)
    channels = 3
    size = height * width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x, img_height, img_width):
    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] - x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

# define function to set layers based on source paper followed
def set_cnn_layers(source='gatys'):
    if source == 'gatys':
        # config from Gatys et al.
        content_layer = 'block5_conv2'
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                        'block4_conv1', 'block5_conv1']
    elif source == 'johnson':
        # config from Johnson et al.
        content_layer = 'block2_conv2'
        style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 
                        'block4_conv3', 'block5_conv3']
    else:
        # use Gatys config as the default anyway
        content_layer = 'block5_conv2'
        style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                        'block4_conv1', 'block5_conv1']
    return content_layer, style_layers

class Evaluator(object):
    def __init__(self, fetch_loss_and_grads, height=None, width=None):
        self.loss_value = None
        self.grads_values = None
        self.fetch_loss_and_grads = fetch_loss_and_grads
        self.height = height
        self.width = width

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.height, self.width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

def style_preview(content, style, output):
    width, height = load_img(content).size
    img_height = height
    img_width = width
    target_image = K.constant(preprocess_image(content, height=img_height, width=img_width))
    style_image = K.constant(preprocess_image(style, height=img_height, width=img_width))

    # Placeholder for our generated image
    generated_image = K.placeholder((1, img_height, img_width, 3))

    # Combine the 3 images into a single batch
    input_tensor = K.concatenate([target_image,
        style_image,generated_image], axis=0)

    # The preview network needs to download to the suitable location
    model = VGG19Model(input_tensor=input_tensor,
                    weights='imagenet',
                    include_top=False,
                    cache_dir=".")

    layers = dict([(layer.name, layer.output) for layer in model.layers])

    # Set the content and style feature layers
    content_layer, style_layers = set_cnn_layers(source=source_paper)

    # initialize total loss
    loss = K.variable(0.)

    # add content loss
    layer_features = layers[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(target_image_features,
                                      combination_features)
    # add style loss
    for layer_name in style_layers:
        layer_features = layers[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features, 
                height=img_height, width=img_width)
        loss += (style_weight / len(style_layers)) * sl

    # add total variation loss
    loss += total_variation_weight * total_variation_loss(generated_image, img_height, img_width)

    # Get the gradients of the generated image wrt the loss
    grads = K.gradients(loss, generated_image)[0]

    # Function to fetch the values of the current loss and the current gradients
    fetch_loss_and_grads = K.function([generated_image], [loss, grads])

    evaluator = Evaluator(fetch_loss_and_grads = fetch_loss_and_grads, height=img_height, width=img_width)

    # Run scipy-based optimization (L-BFGS) over the pixels of the generated image
    # so as to minimize the neural style loss.
    # This is our initial state: the target image.
    # Note that `scipy.optimize.fmin_l_bfgs_b` can only process flat vectors.
    x = preprocess_image(content, height=img_height, width=img_width)
    x = x.flatten()

    for _ in range(iterations):
        x, _, _ = fmin_l_bfgs_b(evaluator.loss, x,
            fprime=evaluator.grads, maxfun=20)
    
    img = x.copy().reshape((img_height, img_width, 3))
    img = deprocess_image(img)
    Image.fromarray(img).save(output, quality=95)

def main():
    content, style, output = getPreviewEnvs()
    parser = ArgumentParser()

    if not os.path.isfile(content):
        parser.error("Content %s does not exist" % content)

    if not os.path.isfile(style):
        parser.error("Style %s does not exist" % style)
    
    style_preview(content=content, style=style, output=output)

if __name__ == '__main__':
    main()

