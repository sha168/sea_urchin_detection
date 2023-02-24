import glob
import os
import shutil

folders = glob.glob('/Users/sha168/Downloads/UIQS/*')

OUT_IMG = "/Users/sha168/Downloads/UIQS/JPEGImages"
OUT_ANN = "/Users/sha168/Downloads/UIQS/Annotations"

os.makedirs(OUT_IMG)
os.makedirs(OUT_ANN)

Annotations = []
for folder in folders:
    annotations = glob.glob(folder + '/Annotations/*.xml')
    for annotation in annotations:
        if annotation not in Annotations:
            try:
                Annotations.append(annotation)

                file_name = annotation.split('Annotations')[-1].split('xml')[0]
                image = annotation.split('Annotations')[0] + 'JPEGImages' + file_name + 'jpg'
                shutil.copy2(annotation, OUT_ANN + file_name + 'xml')
                shutil.copy2(image, OUT_IMG + file_name + 'jpg')
            except:
                print('Skipped ' + annotation)



