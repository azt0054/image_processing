import Queue
import threading
import time

class SimpleThread(threading.Thread):
    def __init__(self, process_data, name, queue):
        threading.Thread.__init__(self)
        self.process_data = process_data
        self.name = name
        self.queue = queue
    def run(self):
        while True:
            data = self.queue.get()
            self.process_data(data)
            self.queue.task_done()


if __name__=='__main__':
    def printer(files):
        print files

    queue = Queue.Queue()
    for i in range(5):
        t = SimpleThread(printer,i,queue)
        t.setDaemon(True)
        t.start()

    for file_name in ["f1","f2","f3"]:
        queue.put(file_name)


    queue.join()
