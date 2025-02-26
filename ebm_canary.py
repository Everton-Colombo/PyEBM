import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_array
from typing import Literal, Optional, List, Tuple
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, Output, Checkbox, HBox, VBox, Label
from IPython.display import display
from itertools import combinations
from tqdm.auto import tqdm

class BaseEBM:
    def __init__(
        self,
        n_cycles: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 1,
        n_bins: int = 256,
        binning_strategy: Literal["quantile", "uniform"] = "quantile",
        smoothing_window: Optional[int] = None,
        max_interactions: int = 10,
        interaction_bins: int = 32,
        fast_bins: int = 8
    ):
        self.n_cycles = n_cycles
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        self.smoothing_window = smoothing_window
        self.max_interactions = max_interactions
        self.interaction_bins = interaction_bins
        self.fast_bins = fast_bins
        
        self.feature_graphs = {}
        self.interaction_graphs = {}
        self.initial_prediction = None
        self.top_interactions = []
        self.feature_names = None
        self.feature_index_map = {}

    def _create_bins(self, values: np.ndarray, sample_weight: Optional[np.ndarray]) -> np.ndarray:
        if self.binning_strategy == "quantile":
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            if sample_weight is not None:
                bin_edges = self._weighted_quantile(values, sample_weight, quantiles)
            else:
                bin_edges = np.quantile(values, quantiles)
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 2:
                bin_edges = np.linspace(values.min(), values.max(), self.n_bins + 1)
        else:
            bin_edges = np.linspace(values.min(), values.max(), self.n_bins + 1)
        return bin_edges

    def _weighted_quantile(self, values: np.ndarray, weights: np.ndarray, quantiles: List[float]) -> np.ndarray:
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        cum_weights_normalized = cum_weights / total_weight
        
        return np.interp(quantiles, cum_weights_normalized, sorted_values)

    def _initialize_feature_graphs(self, X: np.ndarray, sample_weight: Optional[np.ndarray]):
        n_samples, n_features = X.shape
        self.feature_graphs = {}
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            bin_edges = self._create_bins(feature_values, sample_weight)
            bin_contributions = np.zeros(len(bin_edges) - 1)
            self.feature_graphs[feature_idx] = (bin_edges, bin_contributions)

    def _detect_interactions_fast(self, X: np.ndarray, residuals: np.ndarray):
        n_features = X.shape[1]
        interaction_scores = {}

        # Discretize all features first with proper bin clipping
        discretized = {}
        for i in range(n_features):
            bin_edges = np.linspace(X[:, i].min(), X[:, i].max(), self.fast_bins + 1)
            discretized[i] = np.clip(
                np.digitize(X[:, i], bin_edges) - 1,  # Subtract 1 for 0-based indexing
                0, self.fast_bins - 1  # Ensure indices stay within bounds
            )

        # FAST interaction scoring
        for i, j in tqdm(combinations(range(n_features), 2), 
                        desc="Detecting interactions", 
                        total=(n_features*(n_features-1))//2):
            score = self._fast_interaction_score(discretized[i], discretized[j], residuals)
            interaction_scores[(i, j)] = score

        # Select top interactions
        self.top_interactions = sorted(interaction_scores.items(), key=lambda x: -x[1])[:self.max_interactions]
        self.top_interactions = [pair[0] for pair in self.top_interactions]

    def _fast_interaction_score(self, x_i: np.ndarray, x_j: np.ndarray, residuals: np.ndarray) -> float:
        # Create contingency table with proper dimensions
        contingency = np.zeros((self.fast_bins, self.fast_bins))
        valid_mask = (x_i >= 0) & (x_i < self.fast_bins) & (x_j >= 0) & (x_j < self.fast_bins)
        
        np.add.at(contingency, (x_i[valid_mask], x_j[valid_mask]), residuals[valid_mask])
        
        # Compute variance reduction
        total_var = np.var(residuals[valid_mask])
        counts = np.histogram2d(x_i, x_j, bins=self.fast_bins, range=[[0, self.fast_bins], [0, self.fast_bins]])[0]
        counts = np.where(counts == 0, 1, counts)  # Avoid division by zero
        cell_means = contingency / counts
        cond_var = np.nanmean(np.var(cell_means, axis=1))
        
        return total_var - cond_var

    def _fit_interaction_term(self, X: np.ndarray, residuals: np.ndarray, pair: Tuple[int, int]):
        i, j = pair
        X_pair = X[:, [i, j]]
        
        # Fit interaction tree
        tree = DecisionTreeRegressor(max_depth=2, max_leaf_nodes=4)
        tree.fit(X_pair, residuals)
        
        # Create grid for visualization
        x_bins = np.linspace(X[:, i].min(), X[:, i].max(), self.interaction_bins)
        y_bins = np.linspace(X[:, j].min(), X[:, j].max(), self.interaction_bins)
        xx, yy = np.meshgrid(x_bins, y_bins)
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # Predict and store contributions
        contributions = tree.predict(grid).reshape(xx.shape)
        self.interaction_graphs[(i, j)] = (x_bins, y_bins, contributions)

        # Update residuals
        return tree.predict(X_pair) * self.learning_rate

    def _smooth_contributions(self, contributions: np.ndarray, window_size: int) -> np.ndarray:
        if window_size is None or window_size < 3:
            return contributions
        
        pad = window_size // 2
        padded = np.pad(contributions, (pad, pad), mode='edge')
        kernel = np.ones(window_size) / window_size
        return np.convolve(padded, kernel, mode='valid')

    def set_feature_names(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.feature_index_map = {i: name for i, name in enumerate(feature_names)}

class EBMRegressor(BaseEBM):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        X = check_array(X)
        y = np.asarray(y).flatten()
        
        # Initialize base model
        self.initial_prediction = np.average(y, weights=sample_weight)
        predictions = np.full_like(y, self.initial_prediction)
        residuals = y - predictions

        # Fit additive terms
        self._initialize_feature_graphs(X, sample_weight)
        for _ in range(self.n_cycles):
            for feature_idx in range(X.shape[1]):
                X_feature = X[:, feature_idx].reshape(-1, 1)
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X_feature, residuals, sample_weight=sample_weight)
                
                # Update predictions and residuals
                pred = tree.predict(X_feature) * self.learning_rate
                predictions += pred
                residuals = y - predictions
                
                # Update feature graph
                self._update_feature_graph(feature_idx, tree)

        # Detect and fit interactions
        if self.max_interactions > 0:
            self._detect_interactions_fast(X, residuals)
            for pair in self.top_interactions:
                interaction_contribution = self._fit_interaction_term(X, residuals, pair)
                predictions += interaction_contribution
                residuals = y - predictions

        # Smooth contributions
        if self.smoothing_window is not None:
            for feature_idx in self.feature_graphs:
                bin_edges, contributions = self.feature_graphs[feature_idx]
                smoothed = self._smooth_contributions(contributions, self.smoothing_window)
                self.feature_graphs[feature_idx] = (bin_edges, smoothed)
                
        return self

    def _update_feature_graph(self, feature_idx: int, tree: DecisionTreeRegressor):
        bin_edges, contributions = self.feature_graphs[feature_idx]
        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        preds = tree.predict(midpoints.reshape(-1, 1))
        contributions += preds * self.learning_rate

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        # Add single feature contributions
        for feature_idx in self.feature_graphs:
            bin_edges, contributions = self.feature_graphs[feature_idx]
            bin_indices = np.clip(
                np.searchsorted(bin_edges, X[:, feature_idx], side="right") - 1,
                0, len(contributions) - 1
            )
            predictions += contributions[bin_indices]
        
        # Add interaction contributions
        for (i, j) in self.top_interactions:
            x_bins, y_bins, contributions = self.interaction_graphs[(i, j)]
            xi = np.clip(np.searchsorted(x_bins, X[:, i]) - 1, 0, len(x_bins) - 2)
            xj = np.clip(np.searchsorted(y_bins, X[:, j]) - 1, 0, len(y_bins) - 2)
            predictions += contributions[xi, xj]
        
        return predictions

class EBMVisualizer:
    def __init__(self, model: BaseEBM):
        self.model = model
        self.output = Output()
        self.current_fig = None
        
        # Create widgets
        self.plot_type = Dropdown(options=['Single Feature', 'Interaction'], description='Plot Type:')
        self.feature_dropdown = Dropdown(options=self._get_feature_options(), description='Feature:')
        self.interaction_dropdown = Dropdown(options=self._get_interaction_options(), description='Interaction:')
        
        # Set up observers
        self.plot_type.observe(self._update_ui)
        self.feature_dropdown.observe(self._update_plot)
        self.interaction_dropdown.observe(self._update_plot)
        
        # Initial display
        display(VBox([
            self.plot_type,
            HBox([self.feature_dropdown, self.interaction_dropdown]),
            self.output
        ]))
        self._update_ui()

    def _get_feature_options(self) -> List[Tuple[str, int]]:
        return [(self.model.feature_index_map.get(i, f'Feature {i}'), i) 
                for i in self.model.feature_graphs.keys()]

    def _get_interaction_options(self) -> List[Tuple[str, Tuple[int, int]]]:
        return [(f"{self.model.feature_index_map.get(i, i)} & {self.model.feature_index_map.get(j, j)}", (i, j))
                for (i, j) in self.model.top_interactions]

    def _update_ui(self, change=None):
        if self.plot_type.value == 'Single Feature':
            self.feature_dropdown.layout.display = 'flex'
            self.interaction_dropdown.layout.display = 'none'
        else:
            self.feature_dropdown.layout.display = 'none'
            self.interaction_dropdown.layout.display = 'flex'
        self._update_plot()

    def _update_plot(self, change=None):
        with self.output:
            self.output.clear_output(wait=True)
            if self.current_fig:
                plt.close(self.current_fig)
                
            self.current_fig, ax = plt.subplots(figsize=(10, 6))
            
            if self.plot_type.value == 'Single Feature':
                feature_idx = self.feature_dropdown.value
                bin_edges, contributions = self.model.feature_graphs[feature_idx]
                midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                
                ax.plot(midpoints, contributions)
                ax.set_title(f"Feature Contribution - {self.feature_dropdown.label}")
                ax.set_xlabel("Feature Value")
                ax.set_ylabel("Contribution")
                ax.grid(True)
            else:
                interaction = self.interaction_dropdown.value
                x_bins, y_bins, contributions = self.model.interaction_graphs[interaction]
                
                xx, yy = np.meshgrid(x_bins, y_bins)
                contour = ax.contourf(xx, yy, contributions.T, cmap='coolwarm', levels=20)
                plt.colorbar(contour, ax=ax)
                
                ax.set_title(f"Interaction Contribution - {self.interaction_dropdown.label}")
                ax.set_xlabel(self.model.feature_index_map.get(interaction[0], f"Feature {interaction[0]}"))
                ax.set_ylabel(self.model.feature_index_map.get(interaction[1], f"Feature {interaction[1]}"))
            
            plt.tight_layout()
            plt.show()

# Example usage:
# ebm = EBMRegressor(max_interactions=5)
# ebm.fit(X_train, y_train)
# visualizer = EBMVisualizer(ebm)