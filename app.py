from flask import Flask, request, jsonify
from base64 import b64decode
from os.path import basename
from neural_style import neuralstyle
import urllib

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

    return jsonify({'output': 'http://localhost:9090/' + outputPath}) 

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__': 
    app.run(port=9090)