from flask import Flask, request, jsonify
from base64 import b64decode
from os.path import basename
import urllib
from collections import namedtuple
from PIL import Image

import tensorflow as tf
tf.set_random_seed(19)

from neural_style import neuralstyle
from model import cyclegan

app = Flask(__name__)

@app.route('/styleTransfer', methods=['GET']) 
def style_transfer(): 
    contentArg = request.args.get('content')
    styleArg = request.args.get('style')

    iterations = request.args.get('iterations', type=int)

    contentPath = b64decode(contentArg)
    stylePath = b64decode(styleArg)

    contentPath = contentPath.decode('utf-8')
    stylePath = stylePath.decode('utf-8')

    # Download the style to local machine
    styleFileName = urllib.request.urlretrieve(stylePath)[0]

    # Download the content file to local machine
    contentFileName = urllib.request.urlretrieve(contentPath)[0]

    # Construct the output file name
    outputname = basename(contentPath) + '_' + basename(stylePath) + '.png'
    outputPath = 'data/outputs/' + outputname

    args = {"content": contentFileName, "styles": {styleFileName}, "output": outputPath, "iterations": iterations}
    styleOp = neuralstyle(args)
    _, error = styleOp.train()

    # Todo: How to add the custom error information to the response
    if error is not None:
        urllib.request.urlcleanup()
        return error

    # Todo: Clear the temporary style and content files
    urllib.request.urlcleanup()

    return jsonify({'output': 'http://localhost:8000/outputs/' + outputname}) 

@app.route('/artistStyle', methods=['GET'])
def art_style():
    # Get the artist name
    model_dir = None
    style = request.args.get('artist')
    model_dir = './checkpoint/' + style
    
    contentArg = request.args.get('content')
    contentPath = b64decode(contentArg)
    contentPath = contentPath.decode('utf-8')
    content_file = urllib.request.urlretrieve(contentPath)[0]

    im = Image.open(content_file)
    width, height = im.size

    fine_width = width
    if (width % 4) != 0:
        fine_width = width - height % 4

    fine_height = height
    if (height % 4) != 0:
        fine_height = height - height % 4

    output_file = style + basename(contentPath)
    OPTIONS = namedtuple('OPTIONS', 'fine_width fine_height input_nc output_nc\
                              L1_lambda lr use_resnet use_lsgan dataset_dir sample_file checkpoint_dir output_dir \
                              ngf ndf max_size phase direction \
                              beta1 epoch epoch_step batch_size train_size output_file')
    
    args = OPTIONS._make((fine_width, fine_height, 3, 3, 10.0, 0.0002, True, True, '', content_file, model_dir, './data/outputs/',64, 64, 50, 'test', 'BtoA',
                         0.5, 200, 100, 1, 1e8, output_file))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    outputPath = None
    tf.reset_default_graph()
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        outputPath = model.test(args)
    
    # resize the file to the original image size
    img = Image.open(outputPath)
    rsImg = img.resize((width,height), Image.ANTIALIAS)
    rsImg.save(outputPath)

     # Clear the temporary content file
    urllib.request.urlcleanup()

    return jsonify({'output': 'http://localhost:8000/outputs/' + basename(outputPath)})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', '*')
    return response

if __name__ == '__main__':
    app.run(port=9090)