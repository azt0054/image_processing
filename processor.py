import sys
from os import *
from config import *
from helper import *
from core import *
checkDir(processed_image_dir)
checkDir(background_image_dir)
checkDir(backgroundless_image_dir)

def checkIfValidImage(filename):
    if isinstance(filename,str):
        return filename.split(".")[-1] in acceptable_image_type
    else:
        return False

def startProcess(files=[]):
    checkDir(processed_image_dir,background_image_dir,backgroundless_image_dir)
    if not files:
        for (dirpath, dirnames, filenames) in walk(source_image_dir):
            files.extend(filenames)

    filter_image_files = [image_file for image_file in files if checkIfValidImage(image_file) ]

    for image_file in filter_image_files:
        source_file = os.path.join(source_image_dir,image_file)
        cv_filter =  CVFilter(source_file)
        cv_filter.moveBackgroundImages()
        if cv_filter.hasBackground():
            cv_filter.fixBackground()
            #cv_filter.display()
            cv_filter.save()
            log("processed",image_file)
        else:
            log("no background found. skipping",image_file)


if __name__ == '__main__':
    input_files = []
    if len(sys.argv)>1:
        input_files = sys.argv[1:]
    startProcess(input_files)
