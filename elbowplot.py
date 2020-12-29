import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.vq import whiten, kmeans

class ElbowPlot:
    def __init__(self, file_loc):
        self.img = cv2.imread(file_loc)
         
        if self.img is None:
            print ("pls check your file location")

    def resize(self, scale_percent = .50):
        img = self.img
    
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)

        new_size = (width, height)
        scaled_img = cv2.resize(img, new_size)

        return scaled_img

    def initalize_df(self):
        
        scaled_img = self.resize()
        
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

        return image_df
    
    def whiten(self, image_df):
        #whiten the dataset to normalize the data & achieve unit variance 
      
        image_df["scaled_red"] = whiten(image_df["red"])
        image_df["scaled_green"] = whiten(image_df["green"])
        image_df["scaled_blue"] = whiten(image_df["blue"])

        return image_df

    def implement_kmeans(self):

        df = self.initalize_df()
        whiten_df = self.whiten(df)

        distortion = []
        cluster = range(1,7)

        for x in cluster:
            center, d = kmeans(whiten_df[["scaled_red", "scaled_green", "scaled_blue"]], x)
            distortion.append(d)
         
        return cluster, distortion, whiten_df

    def draw_elbow_plot (self, cluster, distortion):
        plt.plot(cluster,distortion)
        plt.xticks(cluster)
        plt.savefig("e_plot.png")
        plt.show()

