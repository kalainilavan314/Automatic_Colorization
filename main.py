import numpy as np
import matplotlib.pyplot as plt
import cv2

# Paths to model files
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
image_path = 'pexels-pixabay-45201.jpg'

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

# Add the cluster centers as 1x1 convolutions to the model
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

# Load the image
bw_image = cv2.imread(image_path)
bw_image = cv2.cvtColor(bw_image, cv2.COLOR_BGR2GRAY)
bw_image = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2RGB)

# Convert the image to a floating point data type and scale to range [0, 1]
normalized = bw_image.astype(np.float32) / 255.0

# Convert the image to Lab color space
lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2Lab)

# Extract the L channel
l_channel = lab_image[:, :, 0]

# Subtract 50 for mean-centering
l_channel -= 50

# Resize the L channel to the network's input size
net_input = cv2.dnn.blobFromImage(l_channel, 1.0, (224, 224), (50, 50, 50))

# Set the input to the network
net.setInput(net_input)

# Perform forward pass to get the predicted ab channels
ab_channels = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize the predicted ab channels to match the original image size
ab_channels = cv2.resize(ab_channels, (bw_image.shape[1], bw_image.shape[0]))

# Concatenate the L channel with the predicted ab channels
lab_output = np.concatenate((l_channel[:, :, np.newaxis], ab_channels), axis=2)

# Convert the Lab image back to RGB
colorized_image = cv2.cvtColor(lab_output, cv2.COLOR_Lab2RGB)

# Scale the resulting image to range [0, 1]
colorized_image = np.clip(colorized_image, 0, 1)

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Black and White Image')
plt.imshow(bw_image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Colorized Image')
plt.imshow(colorized_image)
plt.axis('off')

plt.show()
