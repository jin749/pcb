from .AL import AL
import torch
import numpy as np

from dassl.data.transforms.transforms import build_transform
from dassl.data.data_manager import build_data_loader

from sklearn.cluster import KMeans


class WarmStart(AL):
    def __init__(self, cfg, model, unlabeled_dst, U_index, n_class, device, **kwargs):
        super().__init__(cfg, model, unlabeled_dst, U_index, n_class, **kwargs)
        self.device = device

    def get_features(self):
        self.model.eval()
        unlabeled_features = None
        with torch.no_grad():
            unlabeled_loader = build_data_loader(
                self.cfg,
                data_source=self.unlabeled_set,
                batch_size=self.cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                n_domain=self.cfg.DATALOADER.TRAIN_X.N_DOMAIN,
                n_ins=self.cfg.DATALOADER.TRAIN_X.N_INS,
                tfm=build_transform(self.cfg, is_train=False),
                is_train=False,
            )

            # generate entire unlabeled features set
            for data in unlabeled_loader:
                inputs = data["img"].to(self.device)
                out, features = self.model(inputs, get_feature=True)
                if unlabeled_features is None:
                    unlabeled_features = features
                else:
                    unlabeled_features = torch.cat((unlabeled_features, features), 0)

        return unlabeled_features

    def find_closest_points_to_kmeans_cetroids(self, feartures, n_clusters: int, seed=0) -> list:
        if isinstance(feartures, torch.Tensor):
            feartures = feartures.cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(feartures)
        cluster_labels = kmeans.labels_
        centroid_features = kmeans.cluster_centers_

        centroids_points_idxs = self.find_closest_points(feartures, cluster_labels, centroid_features)
        print(f"Centroids points idxs: {sorted(centroids_points_idxs)}")

        return centroids_points_idxs
    
    def find_closest_points(self, features, cluster_labels, centroid_features):
        closest_points_idxs = []
        for centroid_label, centroid_feature in enumerate(centroid_features):
            same_cluster_idxs = np.where(cluster_labels == centroid_label)[0]
            distances = np.linalg.norm(features[same_cluster_idxs] - centroid_feature, axis=1)
            closest_point_idx = np.argmin(distances)
            closest_points_idxs.append(same_cluster_idxs[closest_point_idx])

        return closest_points_idxs
    
    def select(self, n_query, **kwargs):
        unlabeled_features = self.get_features()
        selected_indices = self.find_closest_points_to_kmeans_cetroids(unlabeled_features, n_query, seed=self.cfg.SEED)

        Q_index = [self.U_index[idx] for idx in selected_indices]

        return Q_index
    