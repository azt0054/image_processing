import sys
from os import *
from shutil import *
from config import *
from helper import *
from core import *
from features.s3_downloader import S3Downloader
from features.database import DatabaseConnection

class Processor:
    def __init__(self):
        self.db = DatabaseConnection("processor")
        self.s3_conn = S3Downloader()
        checkDir(processed_image_dir,background_image_dir,backgroundless_image_dir)

    def checkIfValidImage(self,filename):
        if isinstance(filename,str):
            return filename.split(".")[-1] in acceptable_image_type
        else:
            return False

    def getImageData(self):
        log("Fetching data from DB")
        image_data = []
        query = "SELECT i.data_file FROM images i JOIN products p ON i.imageable_id=p.id WHERE p.deleted_at IS NULL"
        start = 0
        limit = 37000
        while 1:
            try:
                db_data = self.db.run(query,limit,start)
                image_data.extend(db_data)
                start += limit
                if not db_data:
                    break
            except:
                break
        return image_data


    def downloadImage(self,files=[]):
        successfull_download = []
        for filename in files:
            if self.s3_conn.getFile(filename):
                successfull_download.append(filename)
        return successfull_download

    def startProcess(self,files=[],online=False):
        if online:
            image_data = self.getImageData()
            input_files = zip(*image_data)[0]
            total_input_files = len(input_files)
            for file_count in range(0,total_input_files,max_download_count):
                end_limit = file_count+max_download_count
                files = input_files[file_count:end_limit]
                successfull_download = self.downloadImage(files)
                self.runFilter(successfull_download)
                log("Finished count ->",str(end_limit))
                rmtree(s3_image_local_dir)
                checkDir(s3_image_local_dir)
        elif files:
            self.runFilter(files)
        else:
            if not files:
                for (dirpath, dirnames, filenames) in walk(source_image_dir):
                    files.extend(filenames)
            self.runFilter(files)


    def runFilter(self,files=[]):
        filter_image_files = [image_file for image_file in files if self.checkIfValidImage(image_file) ]

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
                #unlink(source_file)
                log("no background found. skipping",image_file)


if __name__ == '__main__':
    po = Processor()
    input_files = []
    online = False
    if len(sys.argv)==1:
        po.startProcess(input_files,online)
    elif len(sys.argv)>1:
        online = sys.argv[-1].lower() in ('on',)
        input_files = sys.argv[1:-1] if online else sys.argv[1:]
        po.startProcess(input_files,online)
