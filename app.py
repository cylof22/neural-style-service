from flask import Flask, request, jsonify, send_file
from base64 import b64decode
from os.path import basename
import urllib
from collections import namedtuple

from imageio import imread, imwrite

from skimage.transform import resize
import tensorflow as tf

from neural_style import neuralstyle
from model import cyclegan
from argparse import ArgumentParser

app = Flask(__name__)

MODEL_DIR = ''
CHECKPOINT_DIR = './checkpoint/'

@app.route('/styleTransfer', methods=['POST']) 
def style_transfer(): 
    styleArg = request.args.get('style')
    stylePath = b64decode(styleArg)
    stylePath = stylePath.decode('utf-8')
    styleFileName = urllib.request.urlretrieve(stylePath)[0]

    iterations = request.args.get('iterations', type=int)    

    # Download the content file to local machine
    contentFile = request.files['content']
    contentPath = './data/contents/' + contentFile.filename
    contentFile.save(contentPath)

    # Construct the output file name
    outputname = basename(stylePath) + '_' + contentFile.filename
    outputPath = './data/outputs/' + outputname

    args = {"content": contentPath, "styles": {styleFileName}, "output": outputPath, "iterations": iterations,
        'network': MODEL_DIR}
    
    styleOp = neuralstyle(args)
    _, error = styleOp.train()

    # Todo: How to add the custom error information to the response
    # Do: the remove file asynchronously
    if error is not None:
        urllib.request.urlcleanup()
        return error

    # Todo: Clear the temporary style and content files
    urllib.request.urlcleanup()

    return send_file(outputPath, mimetype=contentFile.mimetype)

@app.route('/artistStyle/<string:style>', methods=['POST'])
def art_style(style):
    # Get the artist name
    model_dir = None

    model_dir = CHECKPOINT_DIR + style
    
    print(request.files)
    contentFile = request.files['content']
    content_file = './data/contents/' + contentFile.filename
    contentFile.save(content_file)

    print(content_file)
    im = imread(content_file)
    width, height, _  = im.shape

    fine_width = width
    if (width % 4) != 0:
        fine_width = width - width % 4

    fine_height = height
    if (height % 4) != 0:
        fine_height = height - height % 4

    output_file = style + '_' + contentFile.filename
    OPTIONS = namedtuple('OPTIONS', 'fine_width fine_height input_nc output_nc\
                              use_resnet use_lsgan sample_file checkpoint_dir output_dir \
                              ngf ndf phase direction \
                              output_file')
    
    args = OPTIONS._make((fine_width, fine_height, 3, 3, True, True, content_file, model_dir, './data/outputs/',
            64, 64,'test', 'BtoA', output_file))

    gpuOptions = tf.GPUOptions(allow_growth=True)
    tfconfig = tf.ConfigProto(gpu_options=gpuOptions)
    tfconfig.allow_soft_placement = True

    outputPath = None
    tf.reset_default_graph()
    with tf.Session(config=tfconfig) as sess:
        model = cyclegan(sess, args)
        outputPath = model.test(args)
    
    # resize the file to the original image size
    img = imread(outputPath)
    img = resize(img, [width, height])
    imwrite(outputPath, img)

    return send_file(outputPath,  mimetype=contentFile.mimetype)

@app.after_request
def after_request(response):
    # Todo: How to set the control allow origin
    response.headers.add("Access-Control-Allow-Origin", "http://www.elforce.net")
    response.headers.add("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS,POST,PUT")
    response.headers.add("Access-Control-Allow-Headers", "Authorization, enctype")

    return response

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--host',
            dest='host', help='style server host',
            metavar='HOST', default='0.0.0.0', required=False)
    parser.add_argument('--port',
            dest='port', help='style server port',
            metavar='PORT', default='9090', required=False)
    parser.add_argument('--modeldir', 
            dest='modeldir', help='style transfer directory',
            metavar='MODEL', default='./', required=False)
    parser.add_argument('--checkpointdir',
            dest='checkpointdir', help='artist transfer checkpoint director', 
            metavar='CHECKPOINTDIR', default='./checkpoint/',required=False)

    return parser
    
if __name__ == '__main__':
    parser = build_parser()
    options = parser.parse_args()

    MODEL_DIR = options.modeldir
    CHECKPOINT_DIR = options.checkpointdir

    app.run(host=options.host,port=int(options.port))