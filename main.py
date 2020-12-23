import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten, kmeans


#set up image to be read (convert to pixels)
image = cv2.imread("polaroid_palette/imgs/test.png")

r = []
g = []
b = []

for row in image:
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

# implement kmeans method & elbow plot (to determine cluster size)
distortion = []
cluster = range(1,11)

for x in cluster:
    center, d = kmeans(image_df[["scaled_red", "scaled_green", "scaled_blue"]], x)
    distortion.append(d)


plt.plot(cluster, distortion)
plt.xticks(cluster)
plt.savefig("e_plot.png")
plt.show()

#obtain cluster centers (standardized value = actual value/std) (append each color cluster)

cluster_num = 4 #obtained from elbow plot 

cluster_center, n = kmeans(image_df[["scaled_red", "scaled_green", "scaled_blue"]], cluster_num)

r_std, g_std, b_std = image_df[["red", "green", "blue"]].std()


main_colors = []

for x in cluster_center:
    red, green, blue = x 
    scaled_r = (red * r_std) /255
    scaled_g = (green * g_std) /255
    scaled_b = (blue * b_std)/ 255
    main_colors.append((scaled_r, scaled_g, scaled_b)) 

print (main_colors)
plt.imshow([main_colors])
plt.savefig("colors.png")
plt.show() 