from cluster import Cluster
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
import scipy 
import binascii

class Color:
    def __init__(self, file_loc):
        self.cluster_num = Cluster(file_loc)
        self.max, self.whiten_df = self.cluster_num.max_distance()

    def main_colors(self):
        image_df = self.whiten_df 
        cluster_center, dist = kmeans(image_df[["scaled_red", "scaled_green", "scaled_blue"]], int(self.max))

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

        return sorted_main_colors

    def hex(self, sorted_main_colors):
        
        all_colors = []
       
        for rgb in sorted_main_colors:
            c = []
            for color in rgb: 
                c.append(int(color*255))
            c_tuple = tuple(c)
            all_colors.append(c_tuple)
    
      
        hex_colors = [] 
    
        for x in all_colors:
            hashtag = "#"
            hex = '%02x%02x%02x' % x
            color = hashtag + hex[0:7] 
            hex_colors.append(color)
        
        return hex_colors

    def draw(self, sorted_main_colors):
        plt.imshow([sorted_main_colors])
        plt.savefig("colors.png")
        plt.show() 


