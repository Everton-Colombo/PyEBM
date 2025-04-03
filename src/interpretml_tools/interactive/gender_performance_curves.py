from typing import List, Literal

# Data Processing:
import pandas as pd
import numpy as np
from interpret.glassbox._ebm._ebm import EBMModel, ExplainableBoostingClassifier, ExplainableBoostingRegressor
from ..utils import CombinedEBM
from sklearn.metrics import roc_auc_score

# UI:
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from tqdm import tqdm
import mplcursors

from dataclasses import dataclass, field

@dataclass
class ModelPlotGroup:
    id: str
    label: str
    scatter: plt.scatter
    model_type: Literal["baseline", "tocombine", "combined"]
    instances_data: list[dict]
    # [
    #     {
    #         "model": EBMModel,
    #         "weights": list[float],
    #         "metrics": dict
    #     }
    # ]
    additional_group_data: dict = field(default_factory=lambda: {})

class GenericGroupPerformanceAnalyzer:
    PLOT_STYLES = {
        "tocombine": dict(s=100, marker='o', edgecolors='black', zorder=10, color='blue'),
        "baseline": dict(s=50, marker='P', edgecolors='black', zorder=5, color='orange'),
        "combined": dict(s=25, marker='o', zorder=1, alpha=0.6)
    }
    
    def __init__(self, models_to_combine: List[tuple[str, EBMModel]],
                 baseline_models: List[tuple[str, EBMModel]],
                 X_test: pd.DataFrame, y_test: np.ndarray,
                 X_train: pd.DataFrame = None, y_train: np.ndarray = None,
                 male_mask: np.ndarray = None, female_mask: np.ndarray = None,
                 feature_of_interest: str = 'sex',
                 metric: Literal["accuracy", "log_likelihood", "auc"] = "accuracy",
                 **kwargs):
        
        self.models_to_combine = np.array(models_to_combine)
        self.baseline_models = np.array(baseline_models)
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.feature_of_interest = feature_of_interest
        self.metric = metric
        self.kwargs = kwargs
        
        # Create masks for groups
        if male_mask is None or female_mask is None:
            # Only set default values if masks weren't passed
            self.male_mask = X_test[feature_of_interest] == 1
            self.female_mask = X_test[feature_of_interest] == 0
        else:
            # Use the passed masks
            self.male_mask = male_mask
            self.female_mask = female_mask
        
        self.fig = None
        self.ax = None
        self.info_output = widgets.Output()
        
        self.model_plot_groups: list[ModelPlotGroup] = []  # List to store model plot groups
        
    def _combine_models(self, weights: list[float], model_indexes: list[int] = None):
        """Combine models using InterpretML's API capabilities"""
        if model_indexes is None:
            model_indexes = list(range(len(self.models_to_combine)))
        
        selected_models = self.models_to_combine[model_indexes, 1]
        return CombinedEBM(selected_models, weights)
        
    def _evaluate_model(self, model) -> dict:
        """Evaluate model using InterpretML's prediction format"""
        if self.metric == "accuracy":
            y_pred = model.predict(self.X_test)

            return {
                f'male_{self.metric}': np.mean(self.y_test[self.male_mask] == y_pred[self.male_mask]),
                f'female_{self.metric}': np.mean(self.y_test[self.female_mask] == y_pred[self.female_mask]),
                f'overall_{self.metric}': np.mean(self.y_test == y_pred)
            }
        
        elif self.metric == "log_likelihood":
            y_probs = model.predict_proba(self.X_test)
            eps = 1e-10
            
            # For male samples
            male_probs = y_probs[self.male_mask]
            male_true = self.y_test[self.male_mask]
            male_ll = np.mean(np.log(male_probs[range(len(male_true)), male_true] + eps))
            
            # For female samples
            female_probs = y_probs[self.female_mask]
            female_true = self.y_test[self.female_mask]
            female_ll = np.mean(np.log(female_probs[range(len(female_true)), female_true] + eps))
            
            # For all samples
            overall_ll = np.mean(np.log(y_probs[range(len(self.y_test)), self.y_test] + eps))
            
            return {
                f'male_{self.metric}': male_ll,
                f'female_{self.metric}': female_ll,
                f'overall_{self.metric}': overall_ll
            }
        elif self.metric == "auc":
            y_probs = model.predict_proba(self.X_test)[:, 1]  # Get probabilities for class 1
            
            return {
                f'male_{self.metric}': roc_auc_score(self.y_test[self.male_mask], y_probs[self.male_mask]),
                f'female_{self.metric}': roc_auc_score(self.y_test[self.female_mask], y_probs[self.female_mask]),
                f'overall_{self.metric}': roc_auc_score(self.y_test, y_probs)
            }
            
        raise ValueError(f"Unknown metric: {self.metric}")
    
    def _order_models_to_combine(self):
        ## Ordering of models to combine:
        # Finding tocombine ModelPlotGroup
        tocombine_group = next((mpg for mpg in self.model_plot_groups if mpg.id == "tocombine_models"), None)
        if tocombine_group:
            # Get the models to combine
            model_metrics = [instance["metrics"] for instance in tocombine_group.instances_data]
            
            # Order models by proximity
            ordered_indices = [0]  # Start with the first model
            remaining_indices = list(range(1, len(self.models_to_combine)))
            
            while remaining_indices:
                # Get the last ordered model's metrics
                last_model_metrics = model_metrics[ordered_indices[-1]]
                male_metric = last_model_metrics[f'male_{self.metric}']
                female_metric = last_model_metrics[f'female_{self.metric}']
                
                # Find the closest model
                closest_idx = None
                min_distance = float('inf')
                
                for idx in remaining_indices:
                    curr_metrics = model_metrics[idx]
                    curr_male = curr_metrics[f'male_{self.metric}']
                    curr_female = curr_metrics[f'female_{self.metric}']
                    
                    # Euclidean distance in the performance space
                    distance = ((male_metric - curr_male) ** 2 + 
                                (female_metric - curr_female) ** 2) ** 0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        closest_idx = idx
                
                ordered_indices.append(closest_idx)
                remaining_indices.remove(closest_idx)
            
            # Re-order the models_to_combine array
            self.models_to_combine = self.models_to_combine[ordered_indices]
    
    @staticmethod
    def _generate_random_weights(list_size: int, n_lists: int, strategy: Literal["dirichlet", "uniform", "orthant_uniform", "orthant_chi"] = "dirichlet") -> np.ndarray:
        n = n_lists
        d = list_size
        
        if strategy == "dirichlet":
            # Sample from Dirichlet distribution
            weights = np.random.dirichlet(np.ones(list_size), size=n_lists)
        elif strategy == "uniform":
            # Sample from uniform distribution
            weights = np.random.uniform(0.1, 1.0, size=(n_lists, list_size))
            
            # Normalize each row to sum to 1
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = weights / row_sums
        elif strategy == "orthant_uniform":
            # Since the positive orthant is 1/2^d of the hypersphere,
            # we need to use rejection sampling efficiently
            
            weights = []
            total_generated = 0
            
            # We'll generate points in batches to improve efficiency
            batch_size = max(100, n * 2**d)  # Adjusted for dimensionality
            
            while len(weights) < n:
                # Generate a batch of points on the full hypersphere
                candidates = np.random.normal(size=(batch_size, d))
                norms = np.linalg.norm(candidates, axis=1, keepdims=True)
                candidates = candidates / norms
                
                # Filter for points where all coordinates are positive
                mask = np.all(candidates > 0, axis=1)
                positive_points = candidates[mask]
                
                # Add the found positive points to our collection
                weights.extend(positive_points)
                total_generated += batch_size
                
            # Trim to exactly n points
            weights = np.array(weights[:n])
        elif strategy == "orthant_chi":
            # Generate points using the chi distribution for the positive orthant
            weights = np.random.chisquare(df=1, size=(n, d))
            
            # Normalize to place on the unit hypersphere
            norms = np.linalg.norm(weights, axis=1, keepdims=True)
            weights = weights / norms
        
        return weights
    
    def _setup_combination_models(self, n_combinations):
        """Creates the combination groups, combines the models and stores the results"""
        
        ### Setting up combination groups ###
        combination_group_model_indexes: list[list[int]] = [] # list of list of ints containing indexes of models in each group
        combination_group_model_indexes.append(list(range(len(self.models_to_combine)))) # All models group
        
        self._order_models_to_combine()
        # Add models two by two (if possible)
        if len(self.models_to_combine) >= 3:
            for i in range(0, len(self.models_to_combine) - 1):
                combination_group_model_indexes.append(list(range(i, i + 2)))
        
        # Process each group
        for i, group in enumerate(combination_group_model_indexes):
            # Generate weights for the current group
            weight_sets = self._generate_random_weights(len(group), n_combinations, strategy=self.kwargs.get("weight_strategy", "orthant_chi"))
            # Generate the combined models
            instances_data = []
            for weights in tqdm(weight_sets, desc=f"Processing Group {i + 1}/{len(combination_group_model_indexes)}"):
                combined_model = self._combine_models(weights, group)
                model_metrics = self._evaluate_model(combined_model)
                
                instances_data.append({
                    "type": "combined",
                    "model": combined_model,
                    'weights': weights,
                    'metrics': model_metrics
                })
            
            self.model_plot_groups.append(
                ModelPlotGroup(
                    id=f"combination_group_{i}",
                    label=f"Combination ({'ALL' if i == 0 else group})",
                    scatter=None,  # Placeholder for scatter plot
                    model_type="combined",
                    instances_data=instances_data
                )
            )
        
    def _setup_individual_models(self):
        """Creates the individual models and stores the results"""
        # Process models to combine
        tocombine_instances_data = []
        for i, (label, model) in enumerate(self.models_to_combine):
            metrics = self._evaluate_model(model)
            tocombine_instances_data.append({
                "type": "tocombine",
                "model": model,
                'label': label,
                'metrics': metrics
            })
        self.model_plot_groups.append(
            ModelPlotGroup(
                id="tocombine_models",
                label="Models to Combine",
                scatter=None,  # Placeholder for scatter plot
                model_type="tocombine",
                instances_data=tocombine_instances_data
            )
        )
        
        # Process baseline models
        if len(self.baseline_models) == 0:
            return
        baseline_instances_data = []
        for i, (label, model) in enumerate(self.baseline_models):
            metrics = self._evaluate_model(model)
            baseline_instances_data.append({
                "model": model,
                'label': label,
                'metrics': metrics
            })
        self.model_plot_groups.append(
            ModelPlotGroup(
                id="baseline_models",
                label="Baseline Models",
                scatter=None,  # Placeholder for scatter plot
                model_type="baseline",
                instances_data=baseline_instances_data
            )
        )
        
    def _plot_model_groups(self):
        """Plots the combination groups. which should already be setup"""
        # Create plot with adjusted figsize and more space for legend
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        
        # Adjust the subplot to make room for the legend
        plt.subplots_adjust(right=0.75)
        
        for model_plot_group in self.model_plot_groups:
            
            x_values = [instance_data["metrics"][f'male_{self.metric}'] for instance_data in model_plot_group.instances_data]
            y_values = [instance_data["metrics"][f'female_{self.metric}'] for instance_data in model_plot_group.instances_data]
            
            scatter = self.ax.scatter(x_values, y_values, label=model_plot_group.label,
                                    **self.PLOT_STYLES[model_plot_group.model_type])
            model_plot_group.scatter = scatter
        
        
    def generate_plot(self, n_combinations: int = 100):
        self._setup_individual_models()
        self._setup_combination_models(n_combinations)
        
        self._plot_model_groups()
        self._configure_plot()
        self._setup_interactivity()
        
        display(self._create_display())
    
    def _setup_interactivity(self):
        """Add interactive tooltips with model details."""
        
        self._identify_dominated_models()

        # Create cursor for all scatter plots
        cursor = mplcursors.cursor([mpg.scatter for mpg in self.model_plot_groups])

        @cursor.connect("add")
        def on_add(sel):
            with self.info_output:
                clear_output(wait=True)

                # Find which scatter plot was selected
                selected_scatter = sel.artist
                point_index = sel.index
                
                # Find which model plot group this scatter belongs to
                selected_group: ModelPlotGroup = next((mpg for mpg in self.model_plot_groups if mpg.scatter == selected_scatter), None)
                point_instance_data: dict = selected_group.instances_data[point_index]
                
                # Display the model information
                display(self._generate_info_html(selected_group, point_instance_data))
    
    def _generate_info_html(self, selected_group: ModelPlotGroup, instance_data: dict) -> widgets.HTML:
        """Generate HTML for displaying model information"""
        
        if selected_group.model_type == 'combined':
            ws = ', '.join([f"{name}: {w:.2f}" for name, w in zip(self.models_to_combine[:, 0], instance_data['weights'])])
            combination_weights_row = f"<b>Combination Weights:</b> {ws}"
        else:
            combination_weights_row = ""
        
        base_html: str = (
            f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>"
            f"<h3>Info:</h3>"
            f"<b>Group Label:</b> {selected_group.label.title()}<br>"
            f"<b>Model Label:</b> {instance_data.get('label', 'N/A')}<br>"
            f"{combination_weights_row}<br>"
            f"<h3>Performance:</h3>"
            f"<b>Male {self.metric.title()}:</b> {instance_data['metrics'][f'male_{self.metric}']:.3f}<br>"
            f"<b>Female {self.metric.title()}:</b> {instance_data['metrics'][f'female_{self.metric}']:.3f}<br>"
            f"<b>Overall {self.metric.title()}:</b> {instance_data['metrics'][f'overall_{self.metric}']:.3f}<br>"
            "</div>"
        )

        return widgets.HTML(base_html)

    def _configure_plot(self):
        self.ax.set_xlabel(f"Male {self.metric.title().replace('_', ' ')}",
                         fontsize=12)
        self.ax.set_ylabel(f"Female {self.metric.title().replace('_', ' ')}",
                         fontsize=12)
        
        if self.metric == "accuracy":
            self.ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            
        self.ax.grid(True, alpha=0.3)
        self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                        frameon=False, fontsize=9)
    
    def _toggle_group_visibility(self, group: ModelPlotGroup, change):
        group.scatter.set_visible(change['new'])
        self.fig.canvas.draw_idle()
    
    def _identify_dominated_models(self):
        """
        Identify dominated models in the performance space, marking it in each
        group's instances_data.
        """
        
        all_points = []
        for group in self.model_plot_groups:
            for i, instance_data in enumerate(group.instances_data):
                all_points.append({
                    'group': group,
                    'index': i,
                    'male_metric': instance_data['metrics'][f'male_{self.metric}'],
                    'female_metric': instance_data['metrics'][f'female_{self.metric}'],
                    'model_type': group.model_type
                })
        
        for point in all_points:
            is_dominated = False
            for other_point in all_points:
                if point == other_point:
                    continue
                
                if (other_point['male_metric'] >= point['male_metric'] and 
                    other_point['female_metric'] >= point['female_metric'] and
                    (other_point['male_metric'] > point['male_metric'] or 
                     other_point['female_metric'] > point['female_metric'])):
                    # Mark the point as dominated
                    is_dominated = True
                    break
            
            
            point['group'].instances_data[point['index']]['is_dominated'] = is_dominated
                
    def _toggle_dominated_visibility(self, show_dominated: bool):
        """Toggle visibility of dominated models. _identify_dominated_models must be called first."""        
        for group in self.model_plot_groups:
            if group.id == "tocombine_models":
                continue
            
            sizes = group.scatter.get_sizes()
            new_sizes = []
            
            # Store original sizes if not already stored
            if not 'original_sizes' in group.additional_group_data:
                group.additional_group_data["original_sizes"] = sizes.copy() if len(sizes) > 1 else [sizes[0]] * len(group.instances_data)
            
            for i, instance_data in enumerate(group.instances_data):
                if not show_dominated and instance_data.get('is_dominated', False):
                    new_sizes.append(0) # Hide dominated points by setting size to 0 (preserve position)
                else:
                    new_sizes.append(group.additional_group_data["original_sizes"][i])
            
            group.scatter.set_sizes(new_sizes)
            self.fig.canvas.draw_idle() # redraw
    
    def _create_togglevis_checkboxes(self) -> list[widgets.Checkbox]:
        checkbox_widgets = []
        
        dom_cb = widgets.Checkbox(
            value=True,
            description="Dominated Models",
            style={'description_width': 'initial'},
            layout=widgets.Layout(margin='10px 0')
        )
        dom_cb.observe(
            lambda change: self._toggle_dominated_visibility(change['new']),
            names='value'
        )
        checkbox_widgets.append(dom_cb)
        
        for group in self.model_plot_groups:
            checkbox = widgets.Checkbox(
                value=True,
                description=group.label,
                style={'description_width': 'initial'},
                layout=widgets.Layout(margin='5px 0')
            )
            checkbox.observe(lambda change, g=group: self._toggle_group_visibility(g, change), names='value')
            checkbox_widgets.append(checkbox)
        
        return checkbox_widgets
            
    
    def _create_display(self):
        """Create final widget layout with checkboxes for visibility control"""
        # Create checkboxes for plot control
        checkboxes = self._create_togglevis_checkboxes()
        
        # Create the control panel
        control_panel = widgets.VBox([
            widgets.HTML("<b>Model Details:</b>"),
            self.info_output,
            widgets.HTML("<b>Show/Hide Groups:</b>"),
            widgets.VBox(checkboxes, layout={"border": "solid 1px #ccc", "overflow": "hidden scroll", "height": "200px"}),
        ], layout={'width': '300px', 'margin': '0 20px'})
        
        # Make the figure canvas wider to accommodate the legend
        fig_canvas = self.fig.canvas
        fig_canvas.layout.width = '800px'
        
        return widgets.HBox([
            control_panel,
            fig_canvas
        ])