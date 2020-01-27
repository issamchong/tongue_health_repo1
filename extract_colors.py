import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image , ImageEnhance
from matplotlib import style
from sklearn.cluster import KMeans


   

def getcolors(file_name):    
 windowname = 'red'
 cv2.namedWindow(windowname)
 
   
 img=cv2.imread(file_name,1)
 img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
   
 r,g,b = cv2.split(img)
 images=[r,g,b]
    
 thr = 5
 Max=255
   
 ret1, image = cv2.threshold(img, thr, Max, cv2.THRESH_TOZERO)
   
 plt.imshow(image)
 plt.show()
   
 image = image.reshape((image.shape[0] * image.shape[1], 3))
   #converts image into 3 colomns and hiegh*width depth or an array , each list has all values of one color (RGB)
 clt= KMeans(n_clusters=4, init='k-means++', 
            max_iter=100, n_init=1, verbose=0, random_state=3425) #print("clt is",clt) 
 clt.fit(image)
   #take the centers and convert to integers 8 bit and save them in a list
   
 centroids= clt.cluster_centers_.astype("uint8").tolist() 
   #take the centers and convert to integers 8 bit and save them in a list
 #print("centroids are",centroids)
 np.asanyarray(centroids)
 for i in centroids:
    if i[0]==0 and i[1]==0 and i[2]==0:
      centroids.remove(i)
 centroids.sort()     
# print("centroids are",centroids)
# print("lables are",clt.labels_)
   
 numLabels = np.arange(0, len(np.unique(clt.labels_)) )
   # save in this list integer counting from zero to the  number of lables +1
 print(numLabels)
 (hist, _) = np.histogram(clt.labels_, bins = numLabels)#saved in a list
 
	# normalize the histogram, such that it sums to one
 hist = hist.astype("float")
 hist /= hist.sum()
   
 bar = np.zeros((50, 300, 3), dtype = "uint8")
 startX = 0
   
   #this loop attaches to the bar  sequence of rectangles 
 for (percent, color) in zip(hist, centroids):
  # plot the relative percentage of each cluster
  print("percentaje is ",percent)
  print ("color is",color)
  endX = startX + (percent * 300)
  cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),color, -1)
  startX = endX
  if color[0] > 240 and percent > 0.35:
       bacteria=True
  else:
      bacteria=False 

   
   #show the bar after attacing rectangles 

 plt.imshow(bar)
 plt.show()
 plt.imsave("tongue_colors.png",bar)
 return bacteria
