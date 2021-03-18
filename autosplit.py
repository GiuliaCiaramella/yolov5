import os
import random
import shutil

def autosplit(path, weight = (0.7,0.2,0.1)): #train, val, test percentages  #path to dataset folder containing /images and /labels
  print(path)
  path_images=path+"/images/"
  path_images_list=os.listdir(path_images)
  N = len(path_images_list)
  train = round(weight[0] * N)
  test = round(weight[1] * N)
  val =  round(weight[2] * N)


  os.makedirs(path+'/train/images/',exist_ok=True)
  os.makedirs(path+'/train/labels/',exist_ok=True)

  os.makedirs(path+'/test/images/',exist_ok=True)
  os.makedirs(path+'/test/labels/',exist_ok=True)

  os.makedirs(path+'/val/images/',exist_ok=True)
  os.makedirs(path+'/val/labels/',exist_ok=True)

  #training part
  for i in range(0,train+1):
    if not path_images_list:
      break
    image_file_name = random.choice(path_images_list)
    path_images_list.remove(image_file_name)
    shutil.move(path_images+image_file_name, path+'/train/images/'+image_file_name)
    base = os.path.basename(image_file_name)
    base = base[:-4]+'.txt' #creating.txt name
    shutil.move(path+"/labels/"+base,path+"/train/labels/"+base) #moving labels
    
  path_images=path+"/images/"
  path_images_list=os.listdir(path_images)
  print("\nTraining set is split!")

  for i in range(0,test+1):
    if not path_images_list:
      break        
    image_file_name = random.choice(path_images_list)
    path_images_list.remove(image_file_name)
    shutil.move(path_images+image_file_name, path+'/test/images/'+image_file_name)
    base = os.path.basename(image_file_name)
    base = base[:-4]+'.txt' #creating.txt name
    shutil.move(path+"/labels/"+base,path+"/test/labels/"+base) #moving labels

  path_images=path+"/images/"
  path_images_list=os.listdir(path_images)
  print("\nTest set is split!")
  for i in range(0,val+1):
    if not path_images_list:
      break    
    image_file_name = random.choice(path_images_list)
    path_images_list.remove(image_file_name)
    shutil.move(path_images+image_file_name, path+'/val/images/'+image_file_name)
    base = os.path.basename(image_file_name)
    base = base[:-4]+'.txt' #creating.txt name
    shutil.move(path+"/labels/"+base,path+"/val/labels/"+base) #moving labels
  print("\nValidation set is split!")
  print("\nSplitting data is done!")