import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_array
from typing import Literal
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, Output, Checkbox, HBox, VBox, Label
from IPython.display import display
from tqdm.auto import tqdm  # Added import

class BaseEBM:
    def __init__(
        self,
        n_cycles: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 1,
        n_bins: int = 256,
        binning_strategy: Literal["quantile", "uniform"] = "quantile",
        smoothing_window: int = None,
    ):
        self.n_cycles = n_cycles
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.binning_strategy = binning_strategy
        
        if smoothing_window is not None:
            if not isinstance(smoothing_window, int) or smoothing_window <= 0 or smoothing_window % 2 == 0:
                raise ValueError("smoothing_window must be a positive odd integer or None.")
        self.smoothing_window = smoothing_window
        
        self.feature_graphs = {}
        self.initial_prediction = None

    def _initialize_feature_graphs(self, X, sample_weight=None):
        n_samples, n_features = X.shape
        self.feature_graphs = {}
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            
            if self.binning_strategy == "quantile":
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                if sample_weight is not None:
                    bin_edges = self._weighted_quantile(feature_values, sample_weight, quantiles)
                else:
                    bin_edges = np.quantile(feature_values, quantiles)
                
                bin_edges = np.unique(bin_edges)
                if len(bin_edges) < 2:
                    bin_edges = np.linspace(feature_values.min(), feature_values.max(), self.n_bins + 1)
            elif self.binning_strategy == "uniform":
                bin_edges = np.linspace(feature_values.min(), feature_values.max(), self.n_bins + 1)
            
            bin_contributions = np.zeros(len(bin_edges) - 1)
            self.feature_graphs[feature_idx] = (bin_edges, bin_contributions)

    def _weighted_quantile(self, values, weights, quantiles):
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")
        
        cum_weights_normalized = cum_weights / total_weight
        return np.interp(quantiles, cum_weights_normalized, sorted_values)

    def _smooth_contributions(self, contributions, window_size):
        pad = window_size // 2
        padded = np.pad(contributions, (pad, pad), mode='edge')
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names
        self.feature_index_map = {i: name for i, name in enumerate(feature_names)}


class EBMRegressor(BaseEBM):
    
    def __init__(self, n_cycles=100, learning_rate=0.1, max_depth=1, n_bins=256, binning_strategy="quantile", smoothing_window=None):
        super().__init__(n_cycles=n_cycles, learning_rate=learning_rate, max_depth=max_depth, n_bins=n_bins, binning_strategy=binning_strategy, smoothing_window=smoothing_window)
    
    def fit(self, X, y, sample_weight=None):
        X = check_array(X)
        y = np.asarray(y).flatten()
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).flatten()
            if len(sample_weight) != len(y):
                raise ValueError("sample_weight must have same length as y")
        
        # Weighted average for initial prediction
        self.initial_prediction = np.average(y, weights=sample_weight)
        predictions = np.full_like(y, self.initial_prediction, dtype=np.float64)
        residuals = y - predictions

        self._initialize_feature_graphs(X, sample_weight)

        # Added tqdm progress bar
        for _ in tqdm(range(self.n_cycles), desc="EBM round-robin cycles"):
            for feature_idx in range(X.shape[1]):
                X_feature = X[:, feature_idx].reshape(-1, 1)
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X_feature, residuals, sample_weight=sample_weight)
                
                pred = tree.predict(X_feature).flatten() * self.learning_rate
                predictions += pred
                residuals = y - predictions
                
                self._update_feature_graph(feature_idx, tree)

        if self.smoothing_window is not None:
            for feature_idx in self.feature_graphs:
                bin_edges, contributions = self.feature_graphs[feature_idx]
                smoothed = self._smooth_contributions(contributions, self.smoothing_window)
                self.feature_graphs[feature_idx] = (bin_edges, smoothed)
        
        return self

    def _update_feature_graph(self, feature_idx, tree):
        bin_edges, contributions = self.feature_graphs[feature_idx]
        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        preds = tree.predict(midpoints.reshape(-1, 1)).flatten()
        contributions += preds * self.learning_rate

    def predict(self, X):
        X = check_array(X)
        predictions = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)
        for feature_idx in range(X.shape[1]):
            bin_edges, contributions = self.feature_graphs[feature_idx]
            feature_values = X[:, feature_idx]
            bin_indices = np.clip(
                np.searchsorted(bin_edges, feature_values, side="right") - 1,
                0,
                len(contributions) - 1,
            )
            predictions += contributions[bin_indices]
        return predictions


class EBMClassifier(BaseEBM):
    
    def __init__(self, threshold: float = 0.5, n_cycles=100, learning_rate=0.1, max_depth=1, n_bins=256, binning_strategy="quantile", smoothing_window=None):
        super().__init__(n_cycles=n_cycles, learning_rate=learning_rate, max_depth=max_depth, n_bins=n_bins, binning_strategy=binning_strategy, smoothing_window=smoothing_window)
        self.threshold = threshold

    def fit(self, X, y, sample_weight=None):
        X = check_array(X)
        y = np.asarray(y).flatten()
        
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight).flatten()
            if len(sample_weight) != len(y):
                raise ValueError("sample_weight must have same length as y")
        
        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError("EBMClassifier only supports binary classification.")
        self.classes_ = unique_y
        y_encoded = np.where(y == self.classes_[0], 0, 1)
        
        # Weighted average for positive class probability
        if sample_weight is not None:
            pos = np.average(y_encoded, weights=sample_weight)
        else:
            pos = np.mean(y_encoded)
            
        eps = 1e-10
        self.initial_prediction = np.log((pos + eps) / (1 - pos + eps))
        log_odds = np.full_like(y_encoded, self.initial_prediction, dtype=np.float64)
        probabilities = 1 / (1 + np.exp(-log_odds))
        residuals = y_encoded - probabilities

        self._initialize_feature_graphs(X, sample_weight)

        # Added tqdm progress bar
        for _ in tqdm(range(self.n_cycles), desc="EBM round-robin cycles"):
            for feature_idx in range(X.shape[1]):
                X_feature = X[:, feature_idx].reshape(-1, 1)
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X_feature, residuals, sample_weight=sample_weight)
                
                pred = tree.predict(X_feature).flatten() * self.learning_rate
                log_odds += pred
                probabilities = 1 / (1 + np.exp(-log_odds))
                residuals = y_encoded - probabilities
                
                self._update_feature_graph(feature_idx, tree)

        if self.smoothing_window is not None:
            for feature_idx in self.feature_graphs:
                bin_edges, contributions = self.feature_graphs[feature_idx]
                smoothed = self._smooth_contributions(contributions, self.smoothing_window)
                self.feature_graphs[feature_idx] = (bin_edges, smoothed)
        
        return self

    def _update_feature_graph(self, feature_idx, tree):
        bin_edges, contributions = self.feature_graphs[feature_idx]
        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        preds = tree.predict(midpoints.reshape(-1, 1)).flatten()
        contributions += preds * self.learning_rate

    def predict_proba(self, X):
        X = check_array(X)
        log_odds = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)
        for feature_idx in range(X.shape[1]):
            bin_edges, contributions = self.feature_graphs[feature_idx]
            feature_values = X[:, feature_idx]
            bin_indices = np.clip(
                np.searchsorted(bin_edges, feature_values, side="right") - 1,
                0,
                len(contributions) - 1,
            )
            log_odds += contributions[bin_indices]
        probabilities = 1 / (1 + np.exp(-log_odds))
        return np.vstack([1 - probabilities, probabilities]).T

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.threshold
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)
    

class EBMVisualizer:
    def __init__(self, models: list[BaseEBM], model_names: list[str] = None):
        self.models = models if isinstance(models, (list, tuple)) else [models]
        self.model_names = model_names or [f"Model {i+1}" for i in range(len(self.models))]
        self.feature_options = self._get_feature_options()
        self.output = Output()
        self.current_fig = None  # Track active figure

        # Create widgets and UI layout
        self.model_checkboxes = [Checkbox(value=True, description=name) for name in self.model_names]
        self.feature_dropdown = Dropdown(options=self.feature_options, description='Feature:')
        
        # Set up observers
        for cb in self.model_checkboxes:
            cb.observe(self._update_plot, names='value')    
        self.feature_dropdown.observe(self._update_plot, names='value')

        # Initial render
        display(
            HBox([
            VBox([
                self.feature_dropdown, 
                Label("Visible Models:"), 
                VBox(self.model_checkboxes)
            ]),
            self.output
            ])
        )
        self._update_plot()
    
    def _get_feature_options(self):
        # Get common feature indices across all models
        all_features = [set(m.feature_graphs.keys()) for m in self.models]
        common_indices = sorted(list(set.intersection(*all_features)))
        
        # Create (name, index) options
        options = []
        for idx in common_indices:
            # Get name from first model that has feature_names
            name = None
            for model in self.models:
                if hasattr(model, 'feature_names') and model.feature_names is not None:
                    if idx < len(model.feature_names):
                        name = model.feature_names[idx]
                        break
            options.append((name or f"Feature {idx}", idx))
        return options

    def _update_plot(self, change=None):
        with self.output:
            # Clear previous state
            self.output.clear_output(wait=True)
            if self.current_fig:
                plt.close(self.current_fig)  # Explicitly close previous figure

            # Create new figure
            self.current_fig = plt.figure(figsize=(10, 6), num="EBM Feature Contributions")
            ax = plt.gca()
            ax.clear()
            selected_idx = self.feature_dropdown.value
            selected_name = next(name for name, idx in self.feature_options if idx == selected_idx)

            # Predefine colors from a colormap (e.g., tab10) to ensure consistency.
            colors = plt.get_cmap("tab10").colors

            # Plot active models with consistent colors.
            for i, (model, name, checkbox) in enumerate(zip(self.models, self.model_names, self.model_checkboxes)):
                if checkbox.value and selected_idx in model.feature_graphs:
                    bin_edges, contributions = model.feature_graphs[selected_idx]
                    midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                    ax.plot(midpoints, contributions, label=name, marker='o',
                            markersize=4, color=colors[i % len(colors)])

            plt.title(f"Feature Contributions - {selected_name}")
            plt.xlabel("Feature Value")
            plt.ylabel("Contribution")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()  # Render within output context