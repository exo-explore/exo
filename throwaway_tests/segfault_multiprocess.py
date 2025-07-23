import ctypes;
from multiprocessing import Process

def trigger_segfault():
  ctypes.string_at(0)

def subprocess_main(id: int):
    print(f"SUBPROCESS {id}: PROCESS START")
    trigger_segfault()
    print(f"SUBPROCESS {id}: PROCESS END")

def main():
    """This code tests that a master process is not brought down by 
    segfaults that occur in the child processes
    """

    print("MASTER: PROCESS START")
    procs: list[Process] = []
    for i in range(0, 10):
        p = Process(target=subprocess_main, args=(i,))
        procs.append(p)
        p.start()

    print("MASTER: JOINING SUBPROCESSES")
    for p in procs:
        p.join()

    print("MASTER: PROCESS END")

if __name__ == "__main__":
    main()