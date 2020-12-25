
import cv2
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.cluster.vq import whiten, kmeans, vq
import scipy
import binascii



NUM_CLUSTERS = 4


img = cv2.imread('polaroid_palette/imgs/test2.png')

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


cluster_center, dist = kmeans(image_df[["scaled_red", "scaled_green", "scaled_blue"]], NUM_CLUSTERS)

r_std, g_std, b_std = image_df[["red", "green", "blue"]].std()

main_colors = []

for x in cluster_center:
    red, green, blue = x 
    scaled_r = (red * r_std) /255
    scaled_g = (green * g_std) /255
    scaled_b = (blue * b_std)/ 255
    main_colors.append((scaled_r, scaled_g, scaled_b))

vecs, dist = vq(image_df[["scaled_red", "scaled_green", "scaled_blue"]], cluster_center)         # assign codes
counts, bins = scipy.histogram(vecs, len(cluster_center))

dictionary = dict(zip(counts, main_colors))
sorted_main_colors = []

for key, value in sorted(dictionary.items()):
    sorted_main_colors.append(value)

sorted_main_colors = sorted_main_colors[::-1]

plt.imshow([sorted_main_colors])
plt.savefig("colors.png")
plt.show() 




