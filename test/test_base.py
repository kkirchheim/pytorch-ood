import unittest
import os
import sys
import time
import shutil
from pathlib import Path
import multiprocessing

myDir = os.getcwd()
sys.path.append(myDir)
path = Path(myDir)
a=str(path.parent.absolute())
sys.path.append(a)

try:
    from oodtk.dataset.img.imagenet import ImageNetA, ImageNetC, ImageNetO, ImageNetP, ImageNetR
    from oodtk.dataset.img.cifar import CIFAR10P, CIFAR10C, CIFAR100C
    from oodtk.dataset.img.mnistc import MNISTC
except:
    print("Not able to load the modules")

temp_root_folder = a + "/temp"

def refreshTempStorage():
    try:
        # print("Removing temorary directory\n")
        shutil.rmtree(temp_root_folder)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

def checkDownload():
    files = Path(temp_root_folder).glob('*')
    for file in files:
        file_size = os.path.getsize(file)
        if file_size > 0:
            # print(file, " has downloaded : ", file_size, " bytes in 10 seconds")
            return True
        else:
            # print("Issue found with file :", file)
            return False

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

class MyTestCase(unittest.TestCase):
    def test_download_ImageNetA(self):
        #Start
        refreshTempStorage()
        p = multiprocessing.Process(target=downloadSelector, args=(0,))
        p.start()
        time.sleep(3)
        p.terminate()

        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)
    
    def test_download_ImageNetO(self):
        #Start
        refreshTempStorage
        p = multiprocessing.Process(target=downloadSelector, args=(1,))
        p.start()
        time.sleep(3)
        p.terminate()
        
        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_ImageNetR(self):
        #Start
        refreshTempStorage
        p = multiprocessing.Process(target=downloadSelector, args=(2,))
        p.start()
        time.sleep(3)
        p.terminate()
            
        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)
    
    def test_download_ImageNetC(self):
        #Start
        refreshTempStorage
        p = multiprocessing.Process(target=downloadSelector, args=(3,))
        p.start()
        time.sleep(3)
        p.terminate()
        
        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)
    
    def test_download_ImageNetP(self):
        #Start
        refreshTempStorage
        p = multiprocessing.Process(target=downloadSelector, args=(4,))
        p.start()
        time.sleep(3)
        p.terminate()
        
        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_CIFAR10P(self):
        #Start
        refreshTempStorage
        p = multiprocessing.Process(target=downloadSelector, args=(5,))
        p.start()
        time.sleep(3)
        p.terminate()
            
        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_CIFAR10C(self):
        #Start
        p = multiprocessing.Process(target=downloadSelector, args=(6,))
        p.start()
        time.sleep(3)
        p.terminate()
            
        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_CIFAR100C(self):
        #Start
        p = multiprocessing.Process(target=downloadSelector, args=(7,))
        p.start()
        time.sleep(3)
        p.terminate()
            
        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_MNISTC(self):
        #Start
        p = multiprocessing.Process(target=downloadSelector, args=(8,))
        p.start()
        time.sleep(3)
        p.terminate()
            
        # Verify download 
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)
