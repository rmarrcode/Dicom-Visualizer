class GK:
    def __init__(self, n_clusters=4, max_iter=100, m=2, error=1e-6):
        super().__init__()
        self.u, self.centers, self.f = None, None, None
        self.clusters_count = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error

    def fit(self, z):
        N = z.shape[0]
        C = self.clusters_count
        centers = []        
        u = np.random.dirichlet(np.ones(N), size=C)        
        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()            
            centers = self.next_centers(z, u)
            f = self._covariance(z, centers, u)
            dist = self._distance(z, centers, f)
            u = self.next_u(dist)
            iteration += 1            
            # Stopping rule
            if norm(u - u2) < self.error:
                break        
        self.f = f
        self.u = u
        self.centers = centers
        return centers

    def gk_segment(img, clusters=5, smooth=False):    
        # expand dims of binary image (1 channel in z axis)
        new_img = np.expand_dims(img, axis=2)    # reshape
        x, y, z = new_img.shape
        new_img = new_img.reshape(x * y, z)    # segment using GK clustering
        algorithm = GK(n_clusters=clusters)
        cluster_centers = algorithm.fit(new_img)
        output = algorithm.predict(new_img)
        segments = cluster_centers[output].astype(np.int32).reshape(x,y)    
        # get cluster that takes up least space (nodules / airway)
        min_label = min_label_volume(segments)
        segments[np.where(segments != min_label)] = 0
        segments[np.where(segments == min_label)] = 1    
        return segments