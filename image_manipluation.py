import cv2
import matplotlib as plt
import numpy as np
from PIL import Image 
import tkinter as tk
print("all good")


white=np.ones([500,100,3],dtype=np.uint8)*255
img=cv2.imwrite("bg.jpg",white)
bg=Image.fromarray(white)
bg.save("bg2.jpg")
print("all good")





