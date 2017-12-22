#sudo mount -t tmpfs -o size=512m tmpfs /media/ramdisk
#mkdir /media/ramdisk/acado/

print 'You should first modify the path in irepa/python/specpath.'
raise RuntimeError('You should first modify the path in irepa/python/specpath.')
 

import os
ramdiskpath = '/tmp/irepa/'
try: os.mkdir(ramdiskpath)
except: pass
acadoTxtPath = ramdiskpath+'process_%d/'%os.getpid()
try: os.mkdir(acadoTxtPath)
except: pass

acadoBinDir = "/home/nmansard/src/pinocchio/irepa/build/unittest/"

