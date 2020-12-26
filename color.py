import cv2
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.cluster.vq import whiten, kmeans, vq
import scipy
import binascii









import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten, kmeans


#set up image to be read (convert to pixels)
img = cv2.imread('polaroid_palette/imgs/test4.jpg')

scale_percent = .50 
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)

new_size = (width, height)

scaled_img = cv2.resize(img, new_size)



r = []
g = []
b = []

for row in scaled_img:
    for pixel in row:  
        b_channel, g_channel, r_channel = pixel
        r.append(r_channel)
        g.append(g_channel)
        b.append(b_channel)

#set up dataframe 
image_df = pd.DataFrame({"red" : r, 
                          "green" : g, 
                          "blue" : b}, columns=["red", "green", "blue"]) 

#whiten the dataset to normalize the data & achieve unit variance 
image_df["scaled_red"] = whiten(image_df["red"])
image_df["scaled_green"] = whiten(image_df["green"])
image_df["scaled_blue"] = whiten(image_df["blue"])

# # implement kmeans method & elbow plot (to determine cluster size)
distortion = []
cluster = range(1,11)

for x in cluster:
    center, d = kmeans(image_df[["scaled_red", "scaled_green", "scaled_blue"]], x)
    distortion.append(d)


plt.plot(cluster, distortion)
plt.xticks(cluster)
plt.savefig("e_plot.png")
plt.show()