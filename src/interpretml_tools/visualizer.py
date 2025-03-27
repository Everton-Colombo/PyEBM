from typing import Union, Sequence, List
import numpy as np
from ipywidgets import Dropdown, Checkbox, HBox, VBox, Label, Output
from IPython.display import display

import matplotlib.pyplot as plt
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor

class InterpretmlEBMVisualizer:
    def __init__(self, 
                 models: Union[Sequence[ExplainableBoostingClassifier], Sequence[ExplainableBoostingRegressor]],
                 model_names: List[str] = None,
                 feature_names: List[str] = None):
        self.models = models
        self.model_names = model_names or [f"Model {i+1}" for i in range(len(models))]
        self.feature_names = feature_names or models[0].feature_names_in_
        
        # Initialize model colors using tab10 colormap
        self.colors = plt.get_cmap("tab10").colors
        self.model_colors = {i: self.colors[i % len(self.colors)] 
                           for i in range(len(models))}
        
        # Store feature graphs for each model
        self.model_feature_graphs = []
        self._initialize_feature_graphs()
        
        self.feature_options = self._get_common_features()
        self.output = Output()
        self.current_fig = None

        # Create widgets
        self.feature_dropdown = Dropdown(options=self.feature_options, description='Feature:')
        self.model_checkboxes = [Checkbox(value=True, description=name) 
                               for name in self.model_names]
        
        # Set up observers
        self.feature_dropdown.observe(self._update_plot, names='value')
        for cb in self.model_checkboxes:
            cb.observe(self._update_plot, names='value')

    def _initialize_feature_graphs(self):
        """Extract feature graphs for each model"""
        for model in self.models:
            feature_graphs = {}
            for term_idx, term in enumerate(model.term_features_):
                if len(term) == 1:  # Main effect
                    feature_idx = term[0]
                    bin_info = model.bins_[feature_idx]
                    contributions = model.term_scores_[term_idx]
                    
                    # Remove missing (first) and unseen (last) bins
                    contributions = contributions[1:-1]
                    
                    if isinstance(bin_info[0], dict):
                        # Categorical feature
                        categories = list(bin_info[0].keys())
                        bin_edges = categories
                    else:
                        # Continuous feature - use actual data bounds
                        feature_min, feature_max = model.feature_bounds_[feature_idx]
                        cuts = bin_info[0]
                        bin_edges = np.concatenate([
                            [feature_min], 
                            cuts, 
                            [feature_max]
                        ])
                    
                    feature_graphs[feature_idx] = (bin_edges, contributions)
            self.model_feature_graphs.append(feature_graphs)

    def _get_common_features(self):
        """Find features present in all models"""
        common_features = set(self.model_feature_graphs[0].keys())
        for model_graph in self.model_feature_graphs[1:]:
            common_features.intersection_update(model_graph.keys())
        return [(self.feature_names[idx], idx) for idx in sorted(common_features)]

    def show(self):
        """Display interactive visualization"""
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

    def _update_plot(self, change=None):
        with self.output:
            self.output.clear_output(wait=True)
            if self.current_fig:
                plt.close(self.current_fig)
                
            feature_idx = self.feature_dropdown.value
            feature_name = self.feature_names[feature_idx]

            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each selected model's contributions
            for model_idx, (model, checkbox) in enumerate(zip(self.models, self.model_checkboxes)):
                if checkbox.value:
                    color = self.model_colors[model_idx]
                    label = self.model_names[model_idx]
                    
                    # Get model's feature data
                    bin_edges, contributions = self.model_feature_graphs[model_idx][feature_idx]
                    
                    if isinstance(bin_edges, list) and isinstance(bin_edges[0], str):
                        # Categorical - plot bars
                        ax.bar(bin_edges, contributions, alpha=0.5, 
                              label=label, color=color)
                    else:
                        # Continuous - plot lines
                        midpoints = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                        ax.plot(midpoints, contributions, marker='o', markersize=4,
                               label=label, color=color)
                        ax.set_xlim(bin_edges[0], bin_edges[-1])

            plt.title(f"Feature Contributions - {feature_name}")
            plt.xlabel("Feature Value")
            plt.ylabel("Contribution")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()