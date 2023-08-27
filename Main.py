import numpy as np
import cv2
import os

# load the model and image path
prototxt_path='model/colorization_deploy_v2.prototxt'
model_path='model/colorization_release_v2.caffemodel'
kernel_path='model/pts_in_hull.npy'
img_path='rain.jpg'


# load the model
net=cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

points=points.transpose().reshape(2,313,1,1)
net.getLayer(net.getLayerId("class8_ab")).blobs=[points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1,313], 2.606, dtype="float32")]

# LAB
# converting image to tensor for computation
bw_image=cv2.imread(img_path)
normalized=bw_image.astype("float32") / 255.0


lab=cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
# we are converting BGR because the opencv takes BGR not RGB

# resize for same image size and shaope
resized=cv2.resize(lab,(224,224))
# takes l channel of the LAB image
L=cv2.split(resized)[0]
L -= 50


# colorising the image
net.setInput(cv2.dnn.blobFromImage(L))
ab=net.forward()[0, :, :, : ].transpose((1,2,0))

ab=cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
L=cv2.split(lab)[0]

colorized=np.concatenate((L[:,:, np.newaxis], ab), axis=2)
colorized=cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized=(255.0 * colorized).astype("uint8")

cv2.imshow("BW image",bw_image)
cv2.imshow("colorised image",colorized)

# for saving the outup of the program
print("Before saving image:")
print(os.listdir('model/Saved'))
# Filename
filename = 'savedImage2.jpg'
# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()
