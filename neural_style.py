import os
import math

import numpy as np
from PIL import Image
import scipy.misc

from stylize import stylize

def imread(path):
    print(path)
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

class neuralstyle(object):
    def __init__(self, args):
        self.content = args.get('content')
        self.styles = args.get('styles')
        self.network = args.get('network')
        self.output = args.get('output')
        self.iterations = args.get('iterations')
        self.width = args.get('width')
        self.content_weight = args.get('content_weight')
        self.content_weight_blend = args.get('content_weight_blend')
        self.style_weight = args.get('style_weight')
        self.tv_weight = args.get('tv_weight')
        self.style_layer_weight_exp = args.get('style_layer_weight_exp')
        self.learning_rate = args.get('learning_rate')
        self.beta1 = args.get('beta1')
        self.beta2 = args.get('beta2')
        self.epsilon = args.get('epsilon')
        self.style_scales = args.get('style_scales')
        self.style_blend_weights = args.get('style_blend_weights')
        self.initial = args.get('initial')
        self.initial_noiseblend = args.get('initial_noiseblend')
        self.checkpoint_output = args.get('checkpoint_output')
        self.preserve_colors = args.get('preserve_colors')
        self.print_iterations = args.get('print_iterations')
        self.checkpoint_iterations = args.get('checkpoint_iterations')
        self.pooling = args.get('pooling')
        
        if self.beta1 is None:
            self.beta1 = 0.9
        
        if self.beta2 is None:
            self.beta2 = 0.999
        
        if self.epsilon is None:
            self.epsilon = 1e-08

        if self.learning_rate is None:
            self.learning_rate = 1e1

        if self.pooling is None:
            self.pooling = "avg"
        
        if self.network is None:
            self.network = 'imagenet-vgg-verydeep-19.mat'
        else:
            self.network = self.network + 'imagenet-vgg-verydeep-19.mat'

        if self.content_weight is None:
            self.content_weight = 5e0
        
        if self.content_weight_blend is None:
            self.content_weight_blend = 1

        if self.style_weight is None:
            self.style_weight = 5e2
        
        if self.style_scales is None:
            self.style_scales = 1.0

        if self.tv_weight is None:
            self.tv_weight = 1e2
        
        if self.style_layer_weight_exp is None:
            self.style_layer_weight_exp = 1
        

    def train(self):
        # Download the content image to the local machine
        if not os.path.isfile(self.content):
            return None, "Content %s does not exist" % self.content
        
        # Download the style image to the local machine
        if not os.path.isfile(self.network):
            return None, "Network %s does not exist. (Did you forget to download it?)" % self.network
        
        content_image = imread(self.content)
        style_images = [imread(style) for style in self.styles]

        width = self.width
        if width is not None:
            new_shape = (int(math.floor(float(content_image.shape[0]) /
                    content_image.shape[1] * width)), width)
            content_image = scipy.misc.imresize(content_image, new_shape)
        target_shape = content_image.shape
        for i in range(len(style_images)):
            style_scale = self.style_scales
            style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                    target_shape[1] / style_images[i].shape[1])

        style_blend_weights = self.style_blend_weights
        if style_blend_weights is None:
            # default is equal weights
            style_blend_weights = [1.0/len(style_images) for _ in style_images]
        else:
            total_blend_weight = sum(style_blend_weights)
            style_blend_weights = [weight/total_blend_weight
                                for weight in style_blend_weights]

        initial = self.initial
        if initial is not None:
            initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])
            # Initial guess is specified, but not noiseblend - no noise should be blended
            if self.initial_noiseblend is None:
                self.initial_noiseblend = 0.0
        else:
            # Neither inital, nor noiseblend is provided, falling back to random generated initial guess
            if self.initial_noiseblend is None:
                self.initial_noiseblend = 1.0
            if self.initial_noiseblend < 1.0:
                initial = content_image

        if self.checkpoint_output and "%s" not in self.checkpoint_output:
            return None, "To save intermediate images, the checkpoint output parameter must contain `%s` (e.g. `foo%s.jpg`)"
        
        for iteration, image in stylize(
            network=self.network,
            initial=initial,
            initial_noiseblend=self.initial_noiseblend,
            content=content_image,
            styles=style_images,
            preserve_colors=self.preserve_colors,
            iterations=self.iterations,
            content_weight=self.content_weight,
            content_weight_blend=self.content_weight_blend,
            style_weight=self.style_weight,
            style_layer_weight_exp=self.style_layer_weight_exp,
            style_blend_weights=style_blend_weights,
            tv_weight=self.tv_weight,
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            pooling=self.pooling,
            print_iterations=self.print_iterations,
            checkpoint_iterations=self.checkpoint_iterations
        ):
            output_file = None
            combined_rgb = image
            if iteration is not None:
                if self.checkpoint_output:
                    output_file = self.checkpoint_output % iteration
            else:
                output_file = self.output
            if output_file:
                imsave(output_file, combined_rgb)
                return output_file, None
            else:
                return np.clip(combined_rgb, 0, 255).astype(np.uint8), None

        return None, None