from elbowplot import ElbowPlot
import matplotlib.pyplot as plt
import math 


class Cluster:

    def __init__(self, file_loc):
        self.img = ElbowPlot(file_loc)
        self.cluster, self.distortion, self.whiten_df = self.img.implement_kmeans()


    def plot(self):
        plt.plot(self.cluster, self.distortion)
        plt.xticks(self.cluster)
        plt.plot([self.cluster[0], self.cluster[5]], [self.distortion[0], self.distortion[5]])
        plt.show()

    def calc_distance(self, x1, y1, a, b, c):
        #https://www.toppr.com/guides/maths/straight-lines/distance-of-point-from-a-line/#:~:text=Distance%20between%20Two%20Parallel%20Lines,-Two%20lines%20are&text=It%20is%20equal%20to%20the,2%20%2B%20B2)%C2%BD.
       
        d = abs(a*x1 + b*y1 + c)/ math.sqrt(a**2 + b**2)
        return d 

    def max_distance(self):
        a = self.distortion[0] - self.distortion[5]
        b = self.cluster[5] - self.cluster[0]
        c = (self.cluster[0] * self.distortion[5]) - (self.cluster[5] * self.distortion[0])

        all_distances = {}
        for cluster in range (6):
            value = self.calc_distance(self.cluster[cluster], self.distortion[cluster],a, b, c)
            key = cluster + 1

            all_distances[key] = value 

        num_cluster = max(all_distances, key = all_distances.get)

        return num_cluster, self.whiten_df

# test = Cluster("polaroid_palette/imgs/test5.jpg")
# max, whiten_df = test.max_distance()

# print (max)