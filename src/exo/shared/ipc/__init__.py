"""
A set of IPC primitives intended for cross-language use.
Includes things like file-locks, named-pipe duplexes, and so on.

TODO: implement System V IPC primitives??
      1. semaphores w/ SEM_UNDO flag ???
      2. Message Queues => as a replacement for pipe duplexes???
      see: https://www.softprayog.in/programming/system-v-semaphores
           https://tldp.org/LDP/lpg/node21.html
           https://tldp.org/LDP/tlk/ipc/ipc.html
           https://docs.oracle.com/cd/E19683-01/816-5042/auto32/index.html
           https://www.softprayog.in/programming/posix-semaphores

"""
