
from neural_style_preview import style_preview

def art_style_preview_demo():
    contentFileName = "./demo.jpg"
    styleFileName = "BtoA_雪色流光-洛阳市西城区西大街.jpg"
    outputPath = "./testok.jpg"

    style_preview(contentFileName, styleFileName, outputPath)

if __name__ == '__main__':
    art_style_preview_demo()