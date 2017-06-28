import sys
import traceback
import Queue
import threading
from os import *
from shutil import *
from config import *
from helper import *
from core import *
from features.s3_downloader import S3Downloader
from features.database import DatabaseConnection
from features.simple_thread import SimpleThread

class Processor:
    def __init__(self,use_thread=0):
        self.db = DatabaseConnection("processor")
        self.s3_conn = S3Downloader()
        self.filter_queue = None
        self.use_thread = use_thread
        self.processed_count = 0
        self.exception_list=[]
        checkDir(processed_image_dir,background_image_dir,backgroundless_image_dir,comparision_image_dir, \
        uncropped_image_dir,larger_image_dir,test_image_dir)

    def checkIfValidImage(self,filename):
        if isinstance(filename,str):
            return filename.split(".")[-1].lower() in acceptable_image_type
        else:
            return False

    def getImageData(self):
        log("Fetching data from DB")
        image_data = []
        query = "SELECT i.data_file FROM images i JOIN products p ON i.imageable_id=p.id WHERE p.deleted_at IS NULL"
        start = 0
        limit = max_data_count
        target_limt = 200
        while 1:
            try:
                db_data = self.db.run(query,limit,start)
                image_data.extend(db_data)
                start += limit
                if not db_data or start>=target_limt:
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

    def createFilterThread(self,no_of_threads=1):
        self.filter_queue = Queue.Queue()
        for t_id in range(no_of_threads):
            t_id = SimpleThread(self.runFilter,t_id,self.filter_queue)
            t_id.setDaemon(True)
            t_id.start()

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
                for (dirpath, dirnames, filenames) in walk(test_image_dir):#source_image_dir
                    files.extend(filenames)
            if self.use_thread:
                self.createFilterThread(self.use_thread)
                for each_file in files[0:500]:
                    self.filter_queue.put([each_file])
                print "pushed into queue"
                self.filter_queue.join()
            else:
                self.runFilter(files)
        log(self.exception_list)


    def runFilter(self,files=[]):
        filter_image_files = [image_file for image_file in files if self.checkIfValidImage(image_file) ]

        for image_file in filter_image_files:
            try:
                source_file = os.path.join(background_image_dir,image_file)#source_image_dir
                cv_filter =  CVFilter(source_file)
                if cv_filter.hasBackground():
                    test_file = os.path.join(test_image_dir,image_file)
                    copyfile(source_file,test_file)
                    cv_filter.fixBackground()
                    if cv_filter.processed:
                        cv_filter.save()
                        log("processed",image_file)
                """if not cv_filter.is_default and cv_filter.hasBackground():
                    #cv_filter.fixBackground()
                    #cv_filter.display()
                    #cv_filter.save()
                    log("processed",image_file)
                else:
                    #unlink(source_file)
                    log("no background found. skipping",image_file)
                #cv_filter.moveBackgroundImages(0)"""
            except ValueError,e:
                if e.message.startswith("too big"):
                    self.exception_list.append(image_file)
                    log("too big! ->",image_file)
                    larger_file = os.path.join(larger_image_dir,image_file)
                    move(source_file,larger_file)
                else:
                    log("unable to read! ->",image_file)
            except:
                log("unable to read! ->",image_file)
            self.processed_count+=1
            if self.processed_count%10==0:
                log("Processed count ----------------->",str(self.processed_count))


if __name__ == '__main__':
    po = Processor(use_thread=4)
    input_files = []
    online = False
    if len(sys.argv)==1:
        po.startProcess(input_files,online)
    elif len(sys.argv)>1:
        online = sys.argv[-1].lower() in ('on',)
        input_files = sys.argv[1:-1] if online else sys.argv[1:]
        po.startProcess(input_files,online)
