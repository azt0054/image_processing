import sys
import boto
import traceback
from boto.s3.key import Key
import Queue
import threading
import time
from os import *
from shutil import *
from eventlet import GreenPool,patcher
patcher.monkey_patch(all=True)
from config import *
from helper import *
from core import *
from features.database import DatabaseConnection
from features.simple_thread import SimpleThread

class ParallelProcessor:
    def __init__(self):
        self.db = DatabaseConnection("processor")
        self.pool = GreenPool(size=20)
        # connect to the bucket
        self.s3_conn = boto.connect_s3(env('AWS_ACCESS_KEY_ID'),
                        env('AWS_SECRET_ACCESS_KEY'))
        self.bucket = self.s3_conn.get_bucket(s3_bucket_name)
        log("Connecting to S3	->",s3_bucket_name)
        self.bucket_list = None
        self.filter_queue = None
        self.db_queue = None
        self.download_queue = None
        self.processed_count = 0
        checkDir(processed_image_dir,background_image_dir,backgroundless_image_dir,s3_image_local_dir,comparision_image_dir)

    def createFilterThread(self,no_of_threads=1):
        self.filter_queue = Queue.Queue()
        for t_id in range(no_of_threads):
            t_id = SimpleThread(self.runFilter,t_id,self.filter_queue)
            t_id.setDaemon(True)
            t_id.start()

    def createDBThread(self,no_of_threads=1):
        self.db_queue = Queue.Queue()
        for t_id in range(no_of_threads):
            t_id = SimpleThread(self.getImageData,t_id,self.db_queue)
            t_id.setDaemon(True)
            t_id.start()

    def createDownloadThread(self,no_of_threads=1):
        self.download_queue = Queue.Queue()
        for t_id in range(no_of_threads):
            t_id = SimpleThread(self.downloadImage,t_id,self.download_queue)
            t_id.setDaemon(True)
            t_id.start()

    def checkIfValidImage(self,filename):
        if isinstance(filename,str):
            return filename.split(".")[-1].lower() in acceptable_image_type
        else:
            return False

    def getImageData(self,args):
        if len(args)==2:
            start,target_limt = args
        else:
            start = 0
            target_limt = 200
        log("Fetching data from DB")
        image_data = []
        query = "SELECT i.data_file FROM products p LEFT JOIN images i ON (p.id = i.imageable_id AND NOT EXISTS\
                ( SELECT 1 FROM images i2 WHERE i2.imageable_id = p.id AND i2.updated_at > i.updated_at AND i2.imageable_type='Product'))\
                WHERE p.deleted_at IS NULL AND p.approved=True AND i.imageable_type='Product'"
        limit = min(max_data_count,target_limt)
        while 1:
            try:
                db_data = self.db.run(query,limit,start)
                start += limit
                if self.download_queue:
                    log("Fetched count ----------------->",str(start))
                    input_files = zip(*db_data)[0]
                    total_input_files = len(input_files)
                    for file_count in range(0,total_input_files,max_download_count):
                        end_limit = file_count+max_download_count
                        files = input_files[file_count:end_limit]
                        self.download_queue.put(files)
                else:
                    image_data.extend(db_data)
                if not db_data or start>=target_limt:
                    break
            except:
                break
        return image_data

    def saveFile(self,file_obj,file_name):
        try:
            #log("Downloading ->",file_name)
            if os.path.exists(s3_image_local_dir) and not os.path.basename(file_name)=="":
                file_name = os.path.join(s3_image_local_dir,file_name)
                file_obj.get_contents_to_filename(file_name)
            return True
        except:
            log("unable to save")
            traceback.print_exc()
        return False

    def getFile(self, file_name):
        s3_file_name = os.path.join(s3_image_dir,file_name)
        item = self.bucket.get_key(s3_file_name)
        if item:
            if self.saveFile(item,file_name):
                if self.filter_queue:
                    self.filter_queue.put([file_name])
                else:
                    self.runFilter([file_name])
        else:
            log("file not found! ->",file_name)
        return False


    def downloadImage(self,files=[]):
        successfull_download = []
        for filename in files:
            if self.pool and filename:
                self.pool.spawn_n(self.getFile,filename)
            elif not self.pool:
                log("pool not defined!")
            else:
                log("file not defined!")

    def startParallelProcess(self,files=[],online=False):
        if online:
            self.createDBThread()
            self.createFilterThread(20)
            self.createDownloadThread(20)
            if self.db_queue:
                target_limt = 10000
                for start in range(0,target_limt,max_data_count):
                    self.db_queue.put([start,start+max_data_count])
                    log("Finished count ----------------->",str(start+max_data_count))
            else:
                image_data = self.getImageData()
                if not self.download_queue and image_data:
                    input_files = zip(*image_data)[0]
                    total_input_files = len(input_files)
                    for file_count in range(0,total_input_files,max_download_count):
                        end_limit = file_count+max_download_count
                        files = input_files[file_count:end_limit]
                        successfull_download = self.downloadImage(files)
                        if not self.pool and successfull_download:
                            self.runFilter(successfull_download)
                            #rmtree(s3_image_local_dir)
                            #checkDir(s3_image_local_dir)
                        log("Finished count ----------------->",str(end_limit))
            self.filter_queue.join()
            self.db_queue.join()
            self.download_queue.join()
            self.pool.waitall()
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
            try:
                cv_filter =  CVFilter(source_file)
                if not cv_filter.is_default and cv_filter.hasBackground():
                    #cv_filter.fixBackground()
                    #cv_filter.display()
                    #cv_filter.save()
                    #log("processed",image_file)
                    pass
                else:
                    pass
                    #log("no background found. skipping",image_file)
                cv_filter.moveBackgroundImages(0)
            except:
                traceback.print_exc()
                log("file not found. skipping",image_file)
            unlink(source_file)
        self.processed_count+=1
        if self.processed_count%100==0:
            log("Processed count ----------------->",str(self.processed_count))


if __name__ == '__main__':
    po = ParallelProcessor()
    input_files = []
    online = False
    if len(sys.argv)==1:
        po.startParallelProcess(input_files,online)
    elif len(sys.argv)>1:
        online = sys.argv[-1].lower() in ('on',)
        input_files = sys.argv[1:-1] if online else sys.argv[1:]
        po.startParallelProcess(input_files,online)
