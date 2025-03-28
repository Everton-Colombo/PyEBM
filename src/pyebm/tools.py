from .ebm import *

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
    
    def show(self):
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
            
            
class EBMUtils:
    @staticmethod
    def combine_emb_models(models, model_weights):
        model_type = type(models[0])
        if not all(isinstance(m, model_type) for m in models):
            raise ValueError("All models must be of the same type (EBMRegressor or EBMClassifier).")
        
        if len(models) != len(model_weights):
            raise ValueError("Length of models and model_weights must be the same.")
        
        first_model = models[0]
        feature_indices = first_model.feature_graphs.keys()
        
        for model in models[1:]:
            if model.feature_graphs.keys() != feature_indices:
                raise ValueError("All models must have the same feature indices.")
            for idx in feature_indices:
                if not np.allclose(model.feature_graphs[idx][0], first_model.feature_graphs[idx][0], atol=1e-6):
                    raise ValueError(f"Bin edges for feature {idx} do not match across models.")
        
        if isinstance(first_model, EBMRegressor):
            new_model = EBMRegressor(
                n_cycles=first_model.n_cycles,
                learning_rate=first_model.learning_rate,
                max_depth=first_model.max_depth,
                n_bins=first_model.n_bins,
                binning_strategy=first_model.binning_strategy,
                smoothing_window=None
            )
        elif isinstance(first_model, EBMClassifier):
            new_model = EBMClassifier(
                threshold=first_model.threshold,
                n_cycles=first_model.n_cycles,
                learning_rate=first_model.learning_rate,
                max_depth=first_model.max_depth,
                n_bins=first_model.n_bins,
                binning_strategy=first_model.binning_strategy,
                smoothing_window=None
            )
        else:
            raise TypeError("Unsupported model type.")
        
        new_model.initial_prediction = sum(m.initial_prediction * w for m, w in zip(models, model_weights))
        
        new_model.feature_graphs = {}
        for idx in feature_indices:
            bin_edges = first_model.feature_graphs[idx][0]
            combined_contributions = np.zeros_like(first_model.feature_graphs[idx][1])
            for model, weight in zip(models, model_weights):
                contributions = model.feature_graphs[idx][1]
                combined_contributions += contributions * weight
            new_model.feature_graphs[idx] = (bin_edges, combined_contributions)
        
        if hasattr(first_model, 'feature_names') and first_model.feature_names is not None:
            new_model.feature_names = first_model.feature_names.copy()
            new_model.feature_index_map = first_model.feature_index_map.copy()
        
        return new_model


class EBMUtils:
    @staticmethod
    def combine_ebm_models(models, model_weights):
        if not models:
            raise ValueError("No models provided to combine.")
        
        model_type = type(models[0])
        if not all(isinstance(m, model_type) for m in models):
            raise ValueError("All models must be of the same type.")
        
        if len(models) != len(model_weights):
            raise ValueError("Models and weights must have same length.")
        
        first_model = models[0]
        feature_indices = first_model.feature_graphs.keys()
        
        for model in models[1:]:
            if model.feature_graphs.keys() != feature_indices:
                raise ValueError("Models must have identical features.")
            for idx in feature_indices:
                if not np.allclose(model.feature_graphs[idx][0], first_model.feature_graphs[idx][0]):
                    raise ValueError(f"Feature {idx} bin mismatch.")
        
        if isinstance(first_model, EBMRegressor):
            new_model = EBMRegressor(
                n_cycles=first_model.n_cycles,
                learning_rate=first_model.learning_rate,
                max_depth=first_model.max_depth,
                n_bins=first_model.n_bins,
                binning_strategy=first_model.binning_strategy,
                smoothing_window=None
            )
        elif isinstance(first_model, EBMClassifier):
            new_model = EBMClassifier(
                threshold=first_model.threshold,
                n_cycles=first_model.n_cycles,
                learning_rate=first_model.learning_rate,
                max_depth=first_model.max_depth,
                n_bins=first_model.n_bins,
                binning_strategy=first_model.binning_strategy,
                smoothing_window=None
            )
        else:
            raise TypeError("Unsupported model type.")
        
        new_model.initial_prediction = sum(m.initial_prediction * w for m, w in zip(models, model_weights))
        new_model.feature_graphs = {}
        
        for idx in feature_indices:
            bin_edges = first_model.feature_graphs[idx][0]
            combined = np.zeros_like(first_model.feature_graphs[idx][1])
            for model, weight in zip(models, model_weights):
                combined += model.feature_graphs[idx][1] * weight
            new_model.feature_graphs[idx] = (bin_edges, combined)
        
        if hasattr(first_model, 'feature_names'):
            new_model.feature_names = first_model.feature_names.copy()
            new_model.feature_index_map = first_model.feature_index_map.copy()
        
        return new_model
