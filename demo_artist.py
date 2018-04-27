from collections import namedtuple
from glob import glob
import tensorflow as tf
tf.set_random_seed(19)

from os.path import basename
from model import cyclegan
from PIL import Image

def art_style_demo():
    # Get the artist name
    model_dir = './checkpoint/' + 'vangogh2photo_256'
    
    contentFileName = './data/content/929477966.jpg'

    sample_files = glob(contentFileName)
    for sample_file in sample_files:
        im = Image.open(sample_file)
        width, height = im.size

        print(sample_file)
        fine_width = width
        if (width % 4) != 0:
            fine_width = width - width % 4

        fine_height = height
        if (height % 4) != 0:
            fine_height = height - height % 4

        output_file = 'vangogh2photo_256' + basename(contentFileName)
        OPTIONS = namedtuple('OPTIONS', 'fine_width fine_height input_nc output_nc\
                              L1_lambda lr use_resnet use_lsgan dataset_dir sample_file checkpoint_dir output_dir \
                              ngf ndf max_size phase direction \
                              beta1 epoch epoch_step batch_size train_size output_file')

        args = OPTIONS._make((fine_width, fine_height, 3, 3, 10.0, 0.0002, True, True, '', sample_file, model_dir, './data/outputs/',64, 64, 50, 'test', 'BtoA',
                         0.5, 200, 100, 1, 1e8, output_file))

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True

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