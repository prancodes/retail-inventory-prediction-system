import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from src.utils import setup_logging

logger = setup_logging()

def scale_store_features(store_df: pd.DataFrame) -> tuple:
    """
    Standardizes store-level features before clustering.
    Because clustering is distance-based, scaling prevents large-value columns
    from dominating the model.
    """
    logger.info("Scaling store-level features using StandardScaler.")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(store_df)
    return scaled_features, scaler

def run_birch_clustering(scaled_data: np.ndarray, n_clusters: int = 3) -> Birch:
    """
    Applies the BIRCH clustering model on scaled store features.
    """
    logger.info(f"Running BIRCH clustering with n_clusters={n_clusters}")
    birch = Birch(n_clusters=n_clusters)
    birch.fit(scaled_data)
    return birch

def run_gmm_clustering(scaled_data: np.ndarray, n_components: int = 3) -> GaussianMixture:
    """
    Applies Gaussian Mixture Model (GMM) on scaled store features.
    This provides probabilistic (soft) cluster assignments.
    """
    logger.info(f"Running GMM probabilistic clustering with n_components={n_components}")
    # Random state is fixed for reproducibility in results
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(scaled_data)
    return gmm

def evaluate_clustering(scaled_data: np.ndarray, labels: np.ndarray) -> dict:
    """
    Evaluates clustering quality using Silhouette Score and Davies-Bouldin Index.
    """
    # Silhouette score ranges from -1 to 1 (higher is better)
    sil = silhouette_score(scaled_data, labels)
    # Davies-Bouldin score ranges from 0 to infinity (lower is better)
    db = davies_bouldin_score(scaled_data, labels)
    logger.info(f"Clustering Metrics calculated: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}")
    return {"silhouette": sil, "davies_bouldin": db}

def project_to_2d(store_df: pd.DataFrame, scaled_data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Uses PCA to reduce the store features down to 2 dimensions for visualization.
    Returns a DataFrame containing PCA coordinates and their respective cluster labels.
    """
    logger.info("Reducing store features to 2D using PCA.")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'], index=store_df.index)
    pca_df['Cluster'] = labels.astype(str)
    
    logger.info(f"PCA Variance Explained: {pca.explained_variance_ratio_}")
    return pca_df
