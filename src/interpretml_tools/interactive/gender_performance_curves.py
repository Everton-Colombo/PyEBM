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

from dataclasses import dataclass

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

class GenericGroupPerformanceAnalyzer:
    def __init__(self, models_to_combine: List[tuple[str, EBMModel]],
                 baseline_models: List[tuple[str, EBMModel]],
                 X_test: pd.DataFrame, y_test: np.ndarray,
                 X_train: pd.DataFrame = None, y_train: np.ndarray = None,
                 male_mask: np.ndarray = None, female_mask: np.ndarray = None,
                 feature_of_interest: str = 'sex',
                 metric: Literal["accuracy", "log_likelihood", "auc"] = "accuracy"):
        
        self.models_to_combine = list(models_to_combine)  # Changed to list for mutability
        self.baseline_models = list(baseline_models)  # Changed to list for mutability
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.feature_of_interest = feature_of_interest
        self.metric = metric
        
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
        self.status_output = widgets.Output()  # New output for status messages
        self.selection_mode = False  # Track if we're in selection mode
        self.n_combinations = 100  # Store the number of combinations
        
        self.model_plot_groups: list[ModelPlotGroup] = []  # List to store model plot groups
        
    def _combine_models(self, weights: list[float], model_indexes: list[int] = None):
        """Combine models using InterpretML's API capabilities"""
        if model_indexes is None:
            model_indexes = list(range(len(self.models_to_combine)))
        
        selected_models = np.array(self.models_to_combine)[model_indexes, 1]
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
    
    @staticmethod
    def _generate_random_weights(list_size: int, n_lists: int, strategy: Literal["dirichlet", "uniform", "sparse"] = "dirichlet") -> np.ndarray:
        
        if strategy == "dirichlet":
            # Sample from Dirichlet distribution
            weights = np.random.dirichlet(np.ones(list_size), size=n_lists)
        elif strategy == "uniform":
            # Sample from uniform distribution
            weights = np.random.uniform(0.1, 1.0, size=(n_lists, list_size))
            
            # Normalize each row to sum to 1
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = weights / row_sums
        elif strategy == "sparse":
            # Initialize weights array
            weights = np.zeros((n_lists, list_size))
            
            # Ensure we cover all positions being "near-zero" across combinations
            near_zero_value = 0.001  # Very small but not zero to maintain mathematical properties
            
            # Calculate how many combinations per position should have near-zero weights
            samples_per_position = max(1, n_lists // list_size)
            
            for pos in range(list_size):
                # Select random samples to have near-zero weight at this position
                near_zero_samples = np.random.choice(n_lists, size=samples_per_position, replace=False)
                
                # Fill all weights first with random values
                weights[:, pos] = np.random.uniform(0.2, 1.0, size=n_lists)
                
                # Set the selected samples to have near-zero weight at this position
                weights[near_zero_samples, pos] = near_zero_value
            
            # Normalize each row to sum to 1
            row_sums = weights.sum(axis=1, keepdims=True)
            weights = weights / row_sums
        
        return weights
    
    def _setup_combination_models(self, n_combinations):
        """Creates the combination groups, combines the models and stores the results"""
        
        ### Setting up combination groups ###
        combination_group_model_indexes: list[list[int]] = [] # list of list of ints containing indexes of models in each group
        combination_group_model_indexes.append(list(range(len(self.models_to_combine)))) # All models group
        
        
        
        # Add models two by two
        if len(self.models_to_combine) >= 3:
            for i in range(0, len(self.models_to_combine) - 1):
                combination_group_model_indexes.append(list(range(i, i + 2)))
        
        # Process each group
        for i, group in enumerate(combination_group_model_indexes):
            # Generate weights for the current group
            weight_sets = self._generate_random_weights(len(group), n_combinations, strategy="sparse")
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
        PLOT_STYLES = {
            "tocombine": dict(s=100, marker='o', edgecolors='black', zorder=10, color='blue'),
            "baseline": dict(s=50, marker='P', edgecolors='black', zorder=5, color='orange'),
            "combined": dict(s=25, marker='o', zorder=1, alpha=0.6)
        }
        
        # Create plot with adjusted figsize and more space for legend
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Adjust the subplot to make room for the legend
        plt.subplots_adjust(right=0.75)
        
        for model_plot_group in self.model_plot_groups:
            print(f"Plotting group: {model_plot_group.id}")
            
            x_values = [instance_data["metrics"][f'male_{self.metric}'] for instance_data in model_plot_group.instances_data]
            y_values = [instance_data["metrics"][f'female_{self.metric}'] for instance_data in model_plot_group.instances_data]
            
            print(f"x: {x_values}, y: {y_values}")
            
            scatter = self.ax.scatter(x_values, y_values, label=model_plot_group.label,
                                    **PLOT_STYLES[model_plot_group.model_type])
            model_plot_group.scatter = scatter
        
    def _handle_model_selection(self, sel):
        """Handle baseline model selection when in selection mode"""
        selected_scatter = sel.artist
        point_index = sel.index
        
        # Find which model plot group this scatter belongs to
        selected_group = next((mpg for mpg in self.model_plot_groups if mpg.scatter == selected_scatter), None)
        
        # Only process if it's a baseline model and we're in selection mode
        if not self.selection_mode or selected_group.model_type != "baseline":
            return
            
        # Get the selected model instance
        selected_instance = selected_group.instances_data[point_index]
        model_label = selected_instance['label']
        model = selected_instance['model']
        
        # Find the model in baseline_models
        for i, (label, mdl) in enumerate(self.baseline_models):
            if label == model_label and mdl is model:
                # Move the model from baseline to models_to_combine
                self.models_to_combine.append(self.baseline_models.pop(i))
                
                with self.status_output:
                    clear_output(wait=True)
                    print(f"Added '{model_label}' to models to combine. Recalculating...")
                
                # Regenerate the plot
                self._regenerate_plot()
                break
                
    def _regenerate_plot(self):
        """Clear and regenerate the model combinations and plot"""
        # Clear existing plot
        plt.close(self.fig)
        
        # Clear model plot groups
        self.model_plot_groups = []
        
        # Regenerate combinations and plot
        self._setup_combination_models(self.n_combinations)
        self._setup_individual_models()
        self._plot_model_groups()
        self._configure_plot()
        self._setup_interactivity()
        
        # Update the display
        with self.display_container:
            clear_output(wait=True)
            display(self._create_display())
            
    def _toggle_selection_mode(self, change):
        """Toggle the model selection mode"""
        self.selection_mode = change['new']
        with self.status_output:
            clear_output(wait=True)
            if self.selection_mode:
                print("Selection mode ON: Click on a baseline model to add it to models to combine.")
            else:
                print("Selection mode OFF")
    
    def _setup_interactivity(self):
        """Add interactive tooltips with model details."""
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
                selected_group = next((mpg for mpg in self.model_plot_groups if mpg.scatter == selected_scatter), None)
                point_instance_data = selected_group.instances_data[point_index]
                
                # Display the model information
                display(self._generate_info_html(selected_group, point_instance_data))
                
        # Add click handling for model selection
        self.fig.canvas.mpl_connect('button_press_event', lambda event: self._handle_click_event(event, cursor))
        
    def _handle_click_event(self, event, cursor):
        """Handle clicks on the plot"""
        if not self.selection_mode:
            return
            
        # Find the nearest point to the click
        target = None
        min_dist = float('inf')
        
        for mpg in self.model_plot_groups:
            if mpg.model_type != "baseline":
                continue
                
            # Get the x and y data from the scatter plot
            x_data = mpg.scatter.get_offsets()[:, 0]
            y_data = mpg.scatter.get_offsets()[:, 1]
            
            for i, (x, y) in enumerate(zip(x_data, y_data)):
                dist = np.sqrt((event.xdata - x)**2 + (event.ydata - y)**2)
                if dist < min_dist and dist < 0.01:  # Threshold for selection
                    min_dist = dist
                    target = (mpg, i)
        
        if target:
            selected_group, point_index = target
            # Process the selection
            selected_instance = selected_group.instances_data[point_index]
            model_label = selected_instance['label']
            model = selected_instance['model']
            
            # Find the model in baseline_models
            for i, (label, mdl) in enumerate(self.baseline_models):
                if label == model_label and mdl is model:
                    # Move the model from baseline to models_to_combine
                    self.models_to_combine.append(self.baseline_models.pop(i))
                    
                    with self.status_output:
                        clear_output(wait=True)
                        print(f"Added '{model_label}' to models to combine. Recalculating...")
                    
                    # Regenerate the plot
                    self._regenerate_plot()
                    break
        
    def generate_plot(self, n_combinations: int = 100):
        self.n_combinations = n_combinations
        self._setup_combination_models(n_combinations)
        self._setup_individual_models()
        
        self._plot_model_groups()
        self._configure_plot()
        self._setup_interactivity()
        
        # Create a container for the display
        self.display_container = widgets.Output()
        with self.display_container:
            display(self._create_display())
        
        display(self.display_container)
    
    def _generate_info_html(self, selected_group: ModelPlotGroup, instance_data: dict) -> widgets.HTML:
        """Generate HTML for displaying model information"""
        
        if selected_group.model_type == 'combined':
            ws = ', '.join([f"{name}: {w:.2f}" for name, w in zip(np.array(self.models_to_combine)[:, 0], instance_data['weights'])])
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
    
    def _create_togglevis_checkboxes(self) -> list[widgets.Checkbox]:
        checkbox_widgets = []
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
    
    def _hide_dominated_models(self, _=None):
        """Hide models that are dominated by other models on both metrics."""
        with self.status_output:
            clear_output(wait=True)
            print("Hiding dominated models...")
        
        # Get all performance points from all models
        all_points = []
        model_info = []
        
        # Collect points from all groups
        for group in self.model_plot_groups:
            for i, instance in enumerate(group.instances_data):
                # Skip the models_to_combine group as we always want to keep those
                if group.model_type == "tocombine":
                    continue
                    
                metrics = instance["metrics"]
                male_metric = metrics[f'male_{self.metric}']
                female_metric = metrics[f'female_{self.metric}']
                all_points.append((male_metric, female_metric))
                model_info.append((group, i))
        
        # Set to keep track of dominated point indices
        dominated = set()
        
        # Higher values are better for all our metrics
        for i, (x1, y1) in enumerate(all_points):
            for j, (x2, y2) in enumerate(all_points):
                if i != j:
                    if x2 >= x1 and y2 >= y1 and (x2 > x1 or y2 > y1):
                        # Point i is dominated by point j
                        dominated.add(i)
                        break
        
        # Track dominated points by group
        dominated_by_group = {}
        
        for i, (group, point_idx) in enumerate(model_info):
            if i in dominated:
                if group.id not in dominated_by_group:
                    dominated_by_group[group.id] = set()
                dominated_by_group[group.id].add(point_idx)
        
        # Apply visibility to scatter plots
        PLOT_STYLES = {
            "tocombine": dict(s=100, marker='o', edgecolors='black', zorder=10, color='blue'),
            "baseline": dict(s=50, marker='P', edgecolors='black', zorder=5, color='orange'),
            "combined": dict(s=25, marker='o', zorder=1, alpha=0.6)
        }
        
        for group in self.model_plot_groups:
            if group.model_type == "tocombine":
                # Always keep models to combine visible
                continue
                
            if group.id in dominated_by_group:
                # Get original scatter
                old_scatter = group.scatter
                
                # Get the original data
                offsets = old_scatter.get_offsets()
                
                # Create a mask for non-dominated points
                mask = np.ones(len(group.instances_data), dtype=bool)
                for idx in dominated_by_group[group.id]:
                    mask[idx] = False
                    
                if np.any(mask):
                    # Use the original plot style for this group type
                    style = PLOT_STYLES[group.model_type].copy()
                    
                    # Create a new scatter plot with only non-dominated points
                    new_scatter = self.ax.scatter(
                        offsets[mask][:, 0], offsets[mask][:, 1],
                        label=f"{group.label} (non-dominated)",
                        **style
                    )
                    
                    # Hide the original scatter
                    old_scatter.set_visible(False)
                    
                    # Update the group's scatter reference
                    group.scatter = new_scatter
        
        # Update the legend
        self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                      frameon=False, fontsize=9)
        
        # Refresh the plot
        self.fig.canvas.draw_idle()
        
        # Count dominated points
        dominated_count = sum(len(points) for points in dominated_by_group.values())
        
        with self.status_output:
            print(f"Hidden {dominated_count} dominated models out of {len(all_points)} total.")
    
    def _create_display(self):
        """Create final widget layout with checkboxes for visibility control"""
        # Create checkboxes for plot control
        checkboxes = self._create_togglevis_checkboxes()
        
        # Create selection mode toggle
        selection_toggle = widgets.ToggleButton(
            value=False,
            description='Selection Mode',
            disabled=False,
            button_style='', 
            tooltip='Toggle selection mode to add baseline models to the combination',
            layout={'width': 'auto'}
        )
        selection_toggle.observe(self._toggle_selection_mode, names='value')
        
        # Create button to hide dominated models
        hide_dominated_button = widgets.Button(
            description="Hide Dominated Models",
            tooltip="Hide models that are dominated on both metrics",
            layout={'width': 'auto'}
        )
        hide_dominated_button.on_click(self._hide_dominated_models)
        
        # Create the control panel
        control_panel = widgets.VBox([
            widgets.HTML("<b>Model Selection:</b>"),
            selection_toggle,
            widgets.HTML("<b>Optimization:</b>"),
            hide_dominated_button,
            self.status_output,
            widgets.HTML("<b>Model Details:</b>"),
            self.info_output,
            widgets.HTML("<b>Show/Hide Groups:</b>"),
            widgets.VBox(checkboxes)
        ], layout={'width': '300px', 'margin': '0 20px'})
        
        # Make the figure canvas wider to accommodate the legend
        fig_canvas = self.fig.canvas
        fig_canvas.layout.width = '800px'
        
        return widgets.HBox([
            control_panel,
            fig_canvas
        ])