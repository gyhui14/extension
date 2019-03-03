import os
import shutil

root = "./SUN397/"
try:
	os.remove(root + 'ClassName.txt')
	os.remove(root + 'README.txt')
except OSError as e:
	pass

for filename in os.listdir(root):
	for f in os.listdir(root + filename):
		if os.path.isdir(root + filename + "/" + f):
			shutil.move(root + filename + "/" + f, root + f)
		try:
		    os.rmdir(root + filename)
		except OSError as e: 
		   	pass
			
for filename in os.listdir(root):
	for f in os.listdir(root + filename):
		if os.path.isdir(root + filename + "/" + f):
			shutil.move( root + filename + "/" + f, root + filename+ "_" + f)
		try:
		    os.rmdir(root + filename)
		except OSError as e: 
		   	pass