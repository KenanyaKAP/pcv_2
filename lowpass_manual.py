import time
import cv2
import numpy as np 

# Convolution Function
def Convolusi(f,w):
    f = np.float64(f)/255 
    baris = f.shape[0]
    kolom = f.shape[1]
    dkernel = w.shape[0]
    dkernel2= np.int32(np.floor(dkernel/2)) 
    g =np.zeros((baris,kolom,3))
    for y in range(baris):
        for x in range(kolom):
            g[y,x] = 0 
            for i in range(dkernel):
                yy =y+i-dkernel2
                if (yy<0)|(yy>=baris-1):
                    continue 
                for j in range(dkernel):
                    xx =x+j - dkernel2
                    if (xx<0)|(xx>=kolom-1): 
                        continue 
                    g[y,x]=g[y,x]+f[yy,xx]*w[i,j]
    return g
    
# Read image
img = cv2.imread("image.jpg")

# Making kernel
dkernel = 5
kernel = np.ones((dkernel,dkernel))/(dkernel*dkernel)

# Start time
start = time.time()

# Apply convolution
imgResult = Convolusi(img, kernel)

# End time
end = time.time()

print("Execution time: ",(end-start), "second")

# Show images
cv2.imshow('Citra Asli', img)
cv2.imshow('Hasil Convolusi', imgResult)

cv2.waitKey(0)
cv2.destroyAllWindows()
