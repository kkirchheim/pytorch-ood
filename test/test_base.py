import unittest
import os
import sys
import shutil
from pathlib import Path

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
    
class MyTestCase(unittest.TestCase):
    def test_download_ImageNetA(self):
        #Start
        refreshTempStorage()
        ImageNetA(download=True, root=temp_root_folder).base_folder
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)
    
    def test_download_ImageNetC(self):
        #Start
        refreshTempStorage
        ImageNetC(download=True, root=temp_root_folder).base_folder
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)
    
    def test_download_ImageNetP(self):
        #Start
        refreshTempStorage
        ImageNetP(download=True, root=temp_root_folder).base_folder
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_ImageNetO(self):
        #Start
        refreshTempStorage
        ImageNetO(download=True, root=temp_root_folder).base_folder
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_ImageNetR(self):
            #Start
            refreshTempStorage
            ImageNetR(download=True, root=temp_root_folder).base_folder
            if checkDownload:
                self.assertEqual(True, True)
            else:
                self.assertEqual(True, False)

    def test_download_CIFAR10P(self):
        #Start
        refreshTempStorage
        CIFAR10P(download=True, root=temp_root_folder).base_folder
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_CIFAR10C(self):
        #Start
        refreshTempStorage
        CIFAR10C(download=True, root=temp_root_folder).base_folder
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_CIFAR100C(self):
        #Start
        refreshTempStorage
        CIFAR100C(download=True, root=temp_root_folder).base_folder
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)

    def test_download_MNISTC(self):
        #Start
        refreshTempStorage
        MNISTC(download=True, root=temp_root_folder).base_folder
        if checkDownload:
            self.assertEqual(True, True)
        else:
            self.assertEqual(True, False)
