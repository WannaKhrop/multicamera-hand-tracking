from threading import Thread, Event
from time import time
import numpy as np        
        
class SimpleThread(Thread):
    
    def __init__(self, camera_id: int, close_event: Event, target: list[tuple[int, float]]):
        Thread.__init__(self)
        self.id = camera_id
        self.to_close = close_event
        self.capture_target = target

    def get_name(self):
        return 'Camera #{}'.format(self.id)

    def run(self):

        while True:

            number = np.random.normal()
            time_stamp = int(time() * 1000)

            if self.to_close.is_set():
                break

            self.capture_target.append((time_stamp, number))


def main():

    close_threads = Event()
    
    # thread #1
    results_1 = []
    thread_1 = SimpleThread(1, close_threads, results_1)

    # thread #2
    results_2 = []
    thread_2 = SimpleThread(2, close_threads, results_2)

    # thread #3
    results_3 = []
    thread_3 = SimpleThread(3, close_threads, results_3)

    thread_1.start()
    thread_2.start()
    thread_3.start()

    while True:
        data = input()
        if data == 'close':
            close_threads.set()
            break

    print(len(results_1), len(results_2), len(results_3))

if __name__ == '__main__':
    main()