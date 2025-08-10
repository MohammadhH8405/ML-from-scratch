import numpy as np

class Kmean_cluster :
    def __init__(self):
        self.centers = []
        self.n_centers = None
        self.clusters_points = None

    def _euclidean_distance(self, x1, x2) :
        # Compute Euclidean distance between two vectors.
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _silhouette_score(self, x) :
        # Calculate the silhoiette score of clusters to find the best k.
        # recives a list with k lists in it each being a cluster of points.
        # Calculate the score of each point and return the avarage for the models score.
        all_scores = []
        for ind, i in enumerate(x) :
            for j in i :
                a = np.mean([self._euclidean_distance(k, j) for k in x[ind]])
                b_list = []
                for idx, other_cluster in enumerate(x):
                    if idx != ind and other_cluster:
                        b_list.append(np.mean([self._euclidean_distance(j, k) for k in other_cluster]))
                b = min(b_list)
                score = (b-a) / max(a,b)
                all_scores.append(score)
        return np.mean(all_scores)

    def _kmean_build(self, x, k) :
        # Build the kmean model with a certain k.
        min_vals = np.min(x, axis=0)
        max_vals = np.max(x, axis=0)
        best_centers = None
        best_clusters = None
        best_inertia = float('inf')
        # Run the model multiple times with different initial centers 
        # so that we would not get stuck in a local opttimal .
        for _ in range(100) :
            # Start with random points as centers
            centers = np.random.uniform(min_vals, max_vals, size=(k, x.shape[1]))
            old_centers = None
            # Keep running the algo until the centers don't change
            while old_centers is None or not np.allclose(old_centers, centers):
                clusters_points = [[] for i in range(k)]
                for i in x :
                    distances = [self._euclidean_distance(i,center) for center in centers]
                    min_index = distances.index(min(distances))
                    clusters_points[min_index].append(i)
                old_centers = centers.copy()
                centers = [np.mean(cluster, axis=0) if cluster else np.zeros(x.shape[1]) for cluster in clusters_points]
            # Calculate the inertia to find se if we have a more optimal answer
            inertia = sum(
                sum(self._euclidean_distance(p, centers[idx])**2 for p in cluster)
                for idx, cluster in enumerate(clusters_points))

            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_clusters = clusters_points

        return best_centers, best_clusters

    def build(self, x, k=None, n=5) :
        '''
        This method builds the model with a certain k if provided if not will use the 
        silhouette score to find the best k for the points 
        you can set a maximum to the number of clusters with n.
        '''
        if k is None :
            scores = []
            for _ in range(2, n+1) :
                self.n_centers = _
                centers, clusters = self._kmean_build(x, _)
                scores.append(self._silhouette_score(clusters))
            self.n_centers = (scores.index(max(scores))) + 2
            self.centers, self.clusters_points = self._kmean_build(x, self.n_centers)
        else :
            self.n_centers = k
            self.centers, self.clusters_points = self._kmean_build(x, k)

    def predict(self, newx, factor=2.0) :
        """
        This method tells you which cluster your point belongs to and if it is irregular.
        Returns False if point is irregular (too far from its cluster center).
        factor = multiplier of average cluster distance to define 'too far'.
        """
        cluster_ind = [] # Label for each point of the points gien
        regularity = [] # Has False if point is irrigular
        cluster_avg_dist = []
        # Calculate the avrage distance of points and their center in each cluster to find irrigularity.
        for idx, cluster in enumerate(self.clusters_points):
            if cluster:  # avoid empty cluster
                dist_list = [self._euclidean_distance(p, self.centers[idx]) for p in cluster]
                cluster_avg_dist.append(np.mean(dist_list))
            else:
                cluster_avg_dist.append(0)  # no points in cluster

        for i in newx :
            # Calculate the distances from different centers and choose the closest.
            distances = [self._euclidean_distance(i, j) for j in self.centers]
            cluster_ind.append(distances.index(min(distances)))
            distance_center = min(distances)
            # check if point is irrigular
            if distance_center > factor * cluster_avg_dist[cluster_ind[-1]] :
                regularity.append(False)
            else : regularity.append(True)

        return cluster_ind, regularity
