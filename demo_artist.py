from collections import namedtuple
from glob import glob
import tensorflow as tf
tf.set_random_seed(19)

from os.path import basename
from model import cyclegan
from PIL import Image

def art_style_demo():
    # Get the artist name
    model_name = 'cezanne2photo_256'
    model_dir = './checkpoint/' +  model_name
    
    contentFileName = './data/cylo/*'

    sample_files = glob(contentFileName)
    for sample_file in sample_files:
        print(sample_file)
        im = Image.open(sample_file)
        width, height = im.size
        
        fine_width = width
        if (width % 4) != 0:
            fine_width = width - width % 4

        fine_height = height
        if (height % 4) != 0:
            fine_height = height - height % 4

        output_file = model_name + '_' + basename(sample_file)

        print(output_file)
        
        OPTIONS = namedtuple('OPTIONS', 'fine_width fine_height input_nc output_nc\
                              use_resnet use_lsgan sample_file checkpoint_dir output_dir \
                              ngf ndf phase direction \
                              output_file')

        args = OPTIONS._make((fine_width, fine_height, 3, 3, True, True,  sample_file, model_dir, './data/Tim-outputs/', 
            64, 64,'test', 'BtoA', output_file))

        gpuOptions = tf.GPUOptions(allow_growth=True)
        tfconfig = tf.ConfigProto(gpu_options=gpuOptions)
        tfconfig.allow_soft_placement = True

        tf.reset_default_graph()

        outputPath = None
        with tf.Session(config=tfconfig) as sess:
            model = cyclegan(sess, args)
            outputPath = model.test(args)
        
        img = Image.open(outputPath)   
        rsImg = img.resize((width,height), Image.ANTIALIAS)
        rsImg.save(outputPath)

        print(outputPath)

if __name__ == '__main__':
    art_style_demo()