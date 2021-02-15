import threading
import time


class AsyncTaskManager:

    def __init__(self, target, args=(), kwargs=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs if kwargs is not None else {}
        self.condition = threading.Condition()
        self.thread = threading.Thread(target=self.worker)
        self.thread.start()
        self.result = None
        self.stopped = False

    def worker(self):
        while True:
            self.condition.acquire()
            while self.result is not None:
                if self.stopped:
                    self.condition.release()
                    return
                self.condition.notify()
                self.condition.wait()
            self.condition.notify()
            self.condition.release()

            result = (self.target(*self.args, **self.kwargs),)

            self.condition.acquire()
            self.result = result
            self.condition.notify()
            self.condition.release()

    def get_next(self):
        self.condition.acquire()
        while self.result is None:
            self.condition.notify()
            self.condition.wait()
        result = self.result[0]
        self.result = None
        self.condition.notify()
        self.condition.release()
        return result

    def stop(self):
        while self.thread.is_alive():
            self.condition.acquire()
            self.stopped = True
            self.condition.notify()
            self.condition.release()


def task():
    print("begin sleeping...")
    time.sleep(1)
    print("end sleeping.")
    task.i += 1
    print("returns", task.i)
    return task.i


task.i = 0
