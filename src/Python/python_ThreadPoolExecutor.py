'''

python_ThreadPoolExecutor.py

What I want to do is to stop the thread that is hanging and continue with the other threads

'''

import concurrent.futures
import time
import datetime

max_numbers = [10000000, 10000000, 10000000, 10000000, 10000000]

class Task:
    def __init__(self, max_number):
        self.max_number = max_number
        self.interrupt_requested = False

    def __call__(self):
        print("Started:", datetime.datetime.now(), self.max_number)
        last_number = 0;
        for i in range(1, self.max_number + 1):
            if self.interrupt_requested:
                print("Interrupted at", i)
                break
            last_number = i * i
        print("Reached the end")
        return last_number

    def interrupt(self):
        self.interrupt_requested = True


with concurrent.futures.ThreadPoolExecutor(max_workers=len(max_numbers)) as executor:
    tasks = [Task(num) for num in max_numbers]
    for task, future in [(i, executor.submit(i)) for i in tasks]:
        try:
            print(future.result(timeout=1))
        except concurrent.futures.TimeoutError:
            print("this took too long...")
            task.interrupt()


#######
import random
import time
max_numbers = [1,2,3,4,5,6,7,8]
def run_loop(task_number):
    t = random.random()*10
    if t > 5.0:
        print(f"{task_number} I am going to timeout {t:3.1f}")
    else:
        print(f"{task_number} All good")
    time.sleep(t)



class timeTask:
    def __init__(self, task_number):
        self.task_number = task_number
        self.interrupt_requested = False

    def __call__(self):
        # print("Started:", datetime.datetime.now(), self.task_number)
        run_loop(self.task_number)
        print(f"{task.task_number} Reached the end")

    def interrupt(self):
        self.interrupt_requested = True

with concurrent.futures.ThreadPoolExecutor(max_workers=len(max_numbers)) as executor:
    tasks = [timeTask(num) for num in max_numbers]
    for task, future in [(i, executor.submit(i)) for i in tasks]:
        print(f'{task.task_number}')
        try:
            future.result(timeout=5)
            print(f'{task.task_number} ran')
        except concurrent.futures.TimeoutError:
            print(f"{task.task_number} this took too long...")
            task.interrupt()