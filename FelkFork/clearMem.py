from multiprocessing.shared_memory import SharedMemory

name = 'p1' # replace this with the name of your lingering shared memory

shm = SharedMemory(name, create=False)

shm.unlink() # this closes all attachments to the memory and destroys it
print("Cleared memory succesfully")