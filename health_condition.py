from extract_colors import getcolors
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as anim
from PIL import Image , ImageEnhance , ImageDraw,ImageFont
from matplotlib import style
from sklearn.cluster import KMeans
from scipy import misc
#f = misc.face()
#misc.imsave('face.png', f) # uses the Image module (PIL)
#plt.imshow(f)
#plt.show()

fileimage="masked_tongue.jpg"
for i in range(2):  # repeat all process twice in order to get best result
 img = cv2.imread('my tonue.jpg',0)
 #plt.imshow("test",R)
 img2=np.power(img,1)
 img2=cv2.blur(img2,(20,20))
 cv2.imwrite("mask.jpg",img2)
 plt.imshow(img2,cmap="gray")
 mask=Image.open("mask.jpg")
 bright=ImageEnhance.Contrast(mask)
 bright_img=bright.enhance(70)
 bright_img.save("mask.jpg")

imgcolor = cv2.imread('my tonue.jpg')
b,g,r = cv2.split(imgcolor)
cv2.imshow('green', r)
cv2.waitKey(0)
mask = cv2.imread('mask.jpg')
masked_image=cv2.bitwise_and(imgcolor,mask)
cv2.imwrite("masked_tongue.jpg",masked_image)
bacteria=getcolors(fileimage)
print("Bacteria is",bacteria)


plt.show()
edges = cv2.Canny(mask,100,200)
cv2.imwrite("edge.jpg",edges)
img2=cv2.imread("edge.jpg")
width=img2.shape[1]
height=img2.shape[0]
pos=np.array([],np.int8)
print(width,height)
#cv2.line(edges,(0,0),(width,h),(255,255,255),2)
fig=plt.figure()
graph=fig.add_subplot(1,1,1)

for x in range(width):
    for y in range(height):
       y2=(height-1)-y       #To make sure we get the outer pixels only
       pixelcolor=img2[y2,x,1]
       if pixelcolor==255:
           print(y2)
           pos=np.append(pos,[(x,y2)])

pos=pos.reshape(int(len(pos)/2),2)
#cv2.line(edges,(31,344),(31,16),(255,255,255),4)
x=pos[:,0]
y=pos[:,1]
list=[]
#plt.scatter(x,y,s=0.5)

def filter(dat):
  for i in range(len(dat)):
    try:
      d1=dat[i]
      d2=dat[i+1]
      d=d2-d1
      if d in range(-8,8) :
        dat.pop(i)
        dat.pop(i+1)
    except:
        pass  
for i in range(138):
   try: 
    i=i*5
    #d1=x[i]
    #d2=x[i+200]
    #df=d2-d1
    list.append(y[i])
   except:
       pass 
if len(list)>2:
  deform=True
else:
  deform=False
filter(list)
#filter(list)
#filter(list)
edges=abs(edges-255)
cv2.imwrite("edge.jpg",edges)
  


axis=np.arange(0,len(list))
print(list) 

#plt.scatter(axis,list,s=1)

graph.plot(axis,list)
plt.xticks([])
plt.yticks([])
plt.savefig("tongue_shape.png",bbox="tight")
plt.show()
#print(pos)
def put_figure():
 colors_tongue=Image.open("tongue_colors.png")
 shape_tongue=Image.open("tongue_shape.png")
 colors_tongue_copy=colors_tongue.copy()
 shape_tongue_copy=shape_tongue.copy()
 shape_tongue_copy.thumbnail((440,280), Image.ANTIALIAS)
 bg= np.ones((500, 480, 3), dtype = "uint8")
 bg.fill(255)
 plt.imsave("bg.jpg",bg)
 bg=Image.open("bg.jpg")
 txt1 = ImageDraw.Draw(bg)
 font=ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerifBoldItalic.ttf",28,encoding="unic")
 txt1.text((10,10),"Color analisis",fill=(0,0,0),font=font)
 txt2=ImageDraw.Draw(bg)
 txt2.text((10,160),"Shape analisis",fill=(0,0,0),font=font)

 
 final=bg.copy()
 final.paste(colors_tongue_copy,(50,60)) # just specify one corner 
 final.paste(shape_tongue_copy,(10,200)) # just specify one corr
 
  
 plt.imshow(final)
 plt.show()

put_figure()
#another way
#img1=cv2.imread("tongue_colors.png",1)
#img2=cv2.imread("tongue_shape.png",1)
#img3=cv2.imread("bg.jpg",1)
#img4=img3.copy()
#img4[0:50,0:300,:] = img1[0:50,0:300,:] #coordinates is y1 to y2  then x1,x2 part of an image occupied by another 
#img4[0:50,0:300,:] = [0,0,0] #coordinates is y1 to y2  then x1,x2 part of an image occupied by another or replace by a color
#cv2.imshow('Result1', img4)
#cv2.waitKey(0)
