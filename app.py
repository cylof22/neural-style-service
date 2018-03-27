from flask import Flask, request, jsonify
from base64 import b64decode
from os.path import basename
from neural_style import style_transfer_main

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

    outputname = basename(contentPath) + '_' + basename(stylePath) + '.png'
    outputPath = 'data/outputs/' + outputname

    style_transfer_main(contentPath, stylePath, outputPath, iterations)
    return jsonify({'output': 'http://localhost:9090/' + outputPath}) 

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__': 
    app.run()