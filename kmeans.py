"""
Just to remeeber how Kmeans work:
1. assign N number of cluster_centroid randomly ++
2. Assign each point to a cluster_centraoid based on eucledian distance ++
3. Then calculate the centraoid of each cluster
4. Then move each cluster_centroid to regarding cnetroids calculated in step 3
5. Iterate over untill the difference between cluster_centroid and calculated cnetroid have a distance less than epsilon (continue if epsilon is satisfied for iteration on ks)

Note
Point cloud must be a List[List[int]] and the dimension must be whatever user wants, only thing that last dimentsion must be cluster id [x,y,cluster id == -1 for init]
"""

import random

class KMeans():
    def __init__(self, k=1, k_positions=[], epsilon=0.01, point_cloud=[]):
        self.k = k
        self.k_positions = k_positions
        self.epsilon = epsilon
        self.point_cloud = point_cloud

    def random_init_k_positions(self):
        if self.k_positions == []:
            for i in range(self.k):
                random_x = random.randint(1, 5)
                random_y = random.randint(1, 5)
                self.k_positions.append([random_x, random_y])
                print(f"random k inited in: {random_x, random_y}")
            print(f"the initialized ks: {self.k_positions}")

    def assign_each_point_to_k(self):
        for point_index, point in enumerate(self.point_cloud):
            cluster_index = []
            for single_k in self.k_positions:
                eucledian_distance_sum = 0
                for dimension_index, dimension in enumerate(point[:-1]):
                    eucledian_distance_sum += abs(dimension - single_k[dimension_index])**2
                eucledian_distance = eucledian_distance_sum**0.5
                print(f"euclidean distance {eucledian_distance}")
                cluster_index.append(eucledian_distance)

            indice_of_closest_k = cluster_index.index(min(cluster_index))
            print(f"cluster_index: {cluster_index}")
            self.point_cloud[point_index][-1] = indice_of_closest_k

    def gradient_for_each_k(self):
        centroid_points_by_clusters = []
        for indice_of_clusters in range(self.k):
            cluster_points = [point for point in self.point_cloud if point[-1] == indice_of_clusters]
            centroid_points_by_clusters.append(cluster_points)
        return centroid_points_by_clusters

    def calculate_centroid_of_point_for_spesific_k(self):
        centroids = []
        for i in range(self.k):
            cluster_points = [point for point in self.point_cloud if point[-1] == i]
            if not cluster_points:
                centroids.append([0 for _ in range(len(self.point_cloud[0]) - 1)])
                continue
            centroid = []
            for dim in range(len(self.point_cloud[0]) - 1):
                sum_of_dim = sum(point[dim] for point in cluster_points)
                centroid.append(sum_of_dim / len(cluster_points))
            centroids.append(centroid)
        return centroids

    def has_converged(self, new_centroids):
        for old_k, new_k in zip(self.k_positions, new_centroids):
            dist = sum((a - b)**2 for a, b in zip(old_k, new_k))**0.5
            if dist >= self.epsilon:
                return False
        return True

    def fit(self):
        self.random_init_k_positions()
        iteration = 0
        while True:
            print(f"\nIteration {iteration}")
            self.assign_each_point_to_k()
            new_centroids = self.calculate_centroid_of_point_for_spesific_k()
            print(f"New centroids: {new_centroids}")
            if self.has_converged(new_centroids):
                print("Converged.")
                break
            self.k_positions = new_centroids
            iteration += 1


## Function test codes
if __name__ == "__main__":
    kmeans = KMeans(k=2, point_cloud=[[5, 5, -1], [6, 7, -1], [0, 5, -1], [2, 9, -1], [7, 3, -1]])
    kmeans.fit()
    print("\nFinal point assignments:")
    for point in kmeans.point_cloud:
        print(point)
