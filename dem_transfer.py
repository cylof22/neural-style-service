
from neural_style import neuralstyle

def art_style_demo():
    contentFileName = ""
    styleFileName = ""
    outputPath = ""
    iterations = 100
    model = ""

    args = {"content": contentFileName, "styles": {styleFileName}, "output": outputPath, "iterations": iterations,
        'network': model}
    
    styleOp = neuralstyle(args)
    _, error = styleOp.train()

    if error is not None:
        print("transfer error")

if __name__ == '__main__':
    art_style_demo()