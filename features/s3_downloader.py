import boto
import traceback
from boto.s3.key import Key
import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from config import *
from helper import *
checkDir(s3_image_local_dir)


class S3Downloader:
    def __init__(self):
        # connect to the bucket
        self.conn = boto.connect_s3(env('AWS_ACCESS_KEY_ID'),
                        env('AWS_SECRET_ACCESS_KEY'))
        self.bucket = self.conn.get_bucket(s3_bucket_name)
        log("Connecting to S3	->",s3_bucket_name)
        self.bucket_list = None

    def getList(self):
        self.bucket_list = self.bucket.list(s3_image_dir,"")

    def saveList(self):
        s3_list = os.path.join(s3_image_local_dir,"s3_list.txt")
        with open(s3_list) as f:
            for item in self.bucket_list:
                f.write(item.key)

    def saveFile(self,file_obj,file_name):
        try:
            log("Downloading ->",file_name)
            if os.path.exists(s3_image_local_dir) and not os.path.basename(file_name)=="":
                file_name = os.path.join(s3_image_local_dir,file_name)
                file_obj.get_contents_to_filename(file_name)
            return True
        except:
            log("unable to save")
            traceback.print_exc()
        return False

    def getFiles(self,limit=None,step=None):
        for i,item in enumerate(self.bucket_list):
            file_name = str(item.key).split('/')[-1]
            self.saveFile(item,file_name)
            if limit and limit==i:
                break

    def getFile(self, file_name):
        s3_file_name = os.path.join(s3_image_dir,file_name)
        item = self.bucket.get_key(s3_file_name)
        if item:
            if self.saveFile(item,file_name):
                return True
        else:
            log("file not found! ->",file_name)
        return False
