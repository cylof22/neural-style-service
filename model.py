from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *


class cyclegan(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.image_width = args.fine_width
        self.image_height = args.fine_height
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc

        self.discriminator = discriminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS', 'image_width image_height \
                              gf_dim df_dim output_c_dim is_training')
        self.options = OPTIONS._make((args.fine_width, args.fine_height,
                                      args.ngf, args.ndf, args.output_nc,
                                      args.phase == 'train'))

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_width, self.image_height,
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        self.fake_B = self.generator(self.real_A, self.options, False, name="generatorA2B")
        self.fake_A_ = self.generator(self.fake_B, self.options, False, name="generatorB2A")
        self.fake_A = self.generator(self.real_B, self.options, True, name="generatorB2A")
        self.fake_B_ = self.generator(self.fake_A, self.options, True, name="generatorA2B")

        self.DB_fake = self.discriminator(self.fake_B, self.options, reuse=False, name="discriminatorB")
        self.DA_fake = self.discriminator(self.fake_A, self.options, reuse=False, name="discriminatorA")

        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_width, self.image_height,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_width, self.image_height,
                                             self.output_c_dim], name='fake_B_sample')
        self.DB_real = self.discriminator(self.real_B, self.options, reuse=True, name="discriminatorB")
        self.DA_real = self.discriminator(self.real_A, self.options, reuse=True, name="discriminatorA")
        self.DB_fake_sample = self.discriminator(self.fake_B_sample, self.options, reuse=True, name="discriminatorB")
        self.DA_fake_sample = self.discriminator(self.fake_A_sample, self.options, reuse=True, name="discriminatorA")

        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_width, self.image_height,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_width, self.image_height,
                                      self.output_c_dim], name='test_B')
        self.testB = self.generator(self.test_A, self.options, True, name="generatorA2B")
        self.testA = self.generator(self.test_B, self.options, True, name="generatorB2A")

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]
        for var in t_vars: print(var.name)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def test(self, args):
        """Test cyclegan"""
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        which_direction = args.direction
        if which_direction != 'AtoB' and which_direction != 'BtoA':
            print('--which_direction must be AtoB or BtoA')
            return None

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            return None

        out_var, in_var = (self.testB, self.test_A) if which_direction == 'AtoB' else (
            self.testA, self.test_B)
        
        # Processing image
        if args.sample_file is None:
            print("No sample file")
            return None
        
        # Todo: output the same size as the input file
        output_width = args.fine_width
        if output_width is None:
            print("Please set the output image width size")
            return None
        
        output_height = args.fine_height
        if output_height is None:
            print("Please set the output image height size")
            return None
        
        sample_image = [load_test_data(args.sample_file, args.fine_width, args.fine_height)]
        sample_image = np.array(sample_image).astype(np.float32)

        output_dir = args.output_dir
        if output_dir is None:
            output_dir = './'

        image_path = os.path.join(output_dir,
            '{0}_{1}'.format(which_direction, args.output_file))
        fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
        save_images(fake_img, [1, 1], image_path)

        return image_path
