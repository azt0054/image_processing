import psycopg2
import psycopg2.extras
import sys
import os
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from config import *
from helper import *

database_config = {'HOST':env('PG_STAGE_HOST'), 'DATABASE':env('PG_STAGE_DB'), \
                    'USER':env('PG_USER'), 'PASSWORD':env('PG_PASSWORD')}

class DatabaseConnection:
    def __init__(self, name, retrieve=True):
    	conn_string = "host='{HOST}' dbname='{DATABASE}' user='{USER}' password='{PASSWORD}'".format(**database_config)
    	log("Connecting to database	->",database_config['HOST'])
    	self.conn = psycopg2.connect(conn_string)
        self.retrieve = retrieve
        if self.retrieve:
            self.cursor = self.conn.cursor()
        else:
    	    self.cursor = self.conn.cursor('cursor_pg_%s'%name, cursor_factory=psycopg2.extras.DictCursor)

    def run(self,query,limit="",offset=""):
        if limit is not None:
            limit = " LIMIT %s"%limit
        if offset is not None:
            offset = " OFFSET %s"%offset
        query = query + offset + limit
        self.cursor.execute(query)
        if self.retrieve:
            records = self.cursor.fetchall()
            return records

    def display(self):
    	for row_count,row in enumerate(self.cursor,start=1):
    		print "row: %s    %s\n" % (row_count, row)

if __name__ == "__main__":
    query = "SELECT i.data_file FROM images i JOIN products p ON i.imageable_id=p.id WHERE p.deleted_at IS NOT NULL"
    dc = DatabaseConnection("main")
    print dc.run(query,10,0)
    #dc.display()
