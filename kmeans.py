import random

class KMeans():
    def __init__(self, k=1, k_positions=None, epsilon=0.01, point_cloud=None, verbose=True):
        self.k = k
        self.k_positions = k_positions or []
        self.epsilon = epsilon
        self.point_cloud = point_cloud or []
        self.verbose = verbose

    def random_init_k_positions(self):
        if not self.k_positions:
            self.k_positions = [random.choice(self.point_cloud)[:-1] for _ in range(self.k)]
            if self.verbose:
                print(f"Initialized centroids (randomly selected from point cloud): {self.k_positions}")

    def assign_each_point_to_k(self):
        for point_index, point in enumerate(self.point_cloud):
            distances = []
            for centroid in self.k_positions:
                distance = sum((dimension - centroid[dim_idx]) ** 2 for dim_idx, dimension in enumerate(point[:-1])) ** 0.5
                distances.append(distance)
            closest_index = distances.index(min(distances))
            self.point_cloud[point_index][-1] = closest_index
            if self.verbose:
                print(f"Point {point[:-1]} assigned to cluster {closest_index} with distances {distances}")

    def calculate_centroids(self):
        centroids = []
        for i in range(self.k):
            cluster_points = [point for point in self.point_cloud if point[-1] == i]
            if not cluster_points:
                
                centroids.append(random.choice(self.point_cloud)[:-1])
                if self.verbose:
                    print(f"Cluster {i} is empty. Reinitialized with a random point.")
                continue
            centroid = []
            for dim in range(len(self.point_cloud[0]) - 1):
                dim_sum = sum(point[dim] for point in cluster_points)
                centroid.append(dim_sum / len(cluster_points))
            centroids.append(centroid)
        return centroids

    def has_converged(self, new_centroids):
        for old_k, new_k in zip(self.k_positions, new_centroids):
            dist = sum((a - b) ** 2 for a, b in zip(old_k, new_k)) ** 0.5
            if dist >= self.epsilon:
                return False
        return True

    def fit(self):
        self.random_init_k_positions()
        iteration = 0
        while True:
            if self.verbose:
                print(f"\nIteration {iteration}")
            self.assign_each_point_to_k()
            new_centroids = self.calculate_centroids()
            if self.verbose:
                print(f"New centroids: {new_centroids}")
            if self.has_converged(new_centroids):
                if self.verbose:
                    print("Convergence reached. Stopping iterations.")
                break
            self.k_positions = new_centroids
            iteration += 1


if __name__ == "__main__":
    point_cloud = [[5, 5, -1], [6, 7, -1], [0, 5, -1], [2, 9, -1], [7, 3, -1]]
    kmeans = KMeans(k=2, point_cloud=point_cloud, epsilon=0.01, verbose=True)
    kmeans.fit()
    print("\nFinal cluster assignments:")
    for point in kmeans.point_cloud:
        print(point)

