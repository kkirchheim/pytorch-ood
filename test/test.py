import multiprocessing
import os 
import sys
import time
import shutil
from pathlib import Path

try:
    from ..oodtk.dataset.img.imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetP, ImageNetR
    from ..oodtk.dataset.img.cifar import CIFAR10P, CIFAR10C, CIFAR100C
    from ..oodtk.dataset.img.mnistc import MNISTC
except:
    sys.path.insert(0, os.path.join(os.getcwd(),"..","oodtk", "dataset", "img"))
    from imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetP, ImageNetR
    from cifar import CIFAR10P, CIFAR10C, CIFAR100C
    from mnistc import MNISTC


temp_root_folder = "C:\\temp\\temp"


def downloadSelector(select):
    if select==0:
        ImageNetA(download=True, root=temp_root_folder).base_folder
    elif select == 1:
        ImageNetO(download=True, root=temp_root_folder).base_folder
    elif select == 2:
        ImageNetR(download=True, root=temp_root_folder).base_folder
    elif select == 3:
        ImageNetC(download=True, root=temp_root_folder, subset='digital').base_folder
    elif select == 4:
        ImageNetP(download=True, root=temp_root_folder, subset='blur').base_folder
    elif select == 5:
        CIFAR10P(download=True, root=temp_root_folder).base_folder
    elif select == 6:
        CIFAR10C(download=True, root=temp_root_folder).base_folder
    elif select == 7:
        CIFAR100C(download=True, root=temp_root_folder).base_folder
    elif select == 8:
        MNISTC(download=True, root=temp_root_folder, subset='all').base_folder



def checkDowloadFunction():
    total_fucnctions = 9
    # checking whether folder exists or not
    try:
        print("Removing temorary directory\n")
        shutil.rmtree(temp_root_folder)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    procs = []
    for i in range(total_fucnctions):
        print("Adding process", i)
        procs.append(multiprocessing.Process(target=downloadSelector, args=(i,)))
        
    # Start all the dowloading processes    
    for i in range(total_fucnctions):
        procs[i].start()
    
    # Wait for the processe to download few bytes
    time.sleep(10)

    # Terminate all the dowloading processes  
    for i in range(total_fucnctions):
        procs[i].terminate()

    print("\n","#"*100)
    files = Path(temp_root_folder).glob('*')
    for file in files:
        file_size = os.path.getsize(file)
        if file_size > 0:
            print(file, " has downloaded : ", file_size, " bytes in 10 seconds")
        else:
            print("Issue found with file :", file)


if __name__ == '__main__':    
    checkDowloadFunction()