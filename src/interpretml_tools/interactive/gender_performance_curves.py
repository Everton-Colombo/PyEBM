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

class GenericGroupPerformanceAnalyzer:
    def __init__(self, models_to_combine: List[tuple[str, EBMModel]],
                 baseline_models: List[tuple[str, EBMModel]],
                 X_test: pd.DataFrame, y_test: np.ndarray,
                 X_train: pd.DataFrame = None, y_train: np.ndarray = None,
                 male_mask: np.ndarray = None, female_mask: np.ndarray = None,
                 feature_of_interest: str = 'sex',
                 metric: Literal["accuracy", "log_likelihood", "auc"] = "accuracy"):
        
        self.models_to_combine = np.array(models_to_combine)
        self.baseline_models = np.array(baseline_models)
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
        self.scatter_plots = {}  # Dictionary to store scatter plots by group
        self.info_output = widgets.Output()
        self.metrics_data = []
        self.combination_groups = []
        self.group_data = {}  # Dictionary to store data by group
        self.individual_models_data = {}  # Store data for baseline and individual model

    def _combine_models(self, weights: list[float]) -> ExplainableBoostingClassifier:
        """Combine models using InterpretML's API capabilities"""
        return CombinedEBM(self.models_to_combine[:, 1], weights)
        
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

    def _plot_basecomb_models(self):
        """Plots baseline models and models to combine differently, for clarity."""
        # Define a unified style for baseline models
        baseline_style = dict(s=50, marker='P', edgecolors='black', zorder=10, color='orange')
        tocombine_style = dict(s=100, marker='o', edgecolors='black', zorder=10, color='red')

        x_values = []
        y_values = []
        
        # Plot baseline models
        for i, (label, model) in enumerate(self.baseline_models):
            metrics = self._evaluate_model(model)
            x_values.append(metrics[f'male_{self.metric}'])
            y_values.append(metrics[f'female_{self.metric}'])

            # Store metrics for interactivity
            model_id = f"baseline_{i}"
            self.individual_models_data[model_id] = {
                'label': label,
                'type': 'baseline',
                'metrics': metrics
            }

        # Create a single scatter plot for all baseline models
        scatter = self.ax.scatter(x_values, y_values, label="Baseline Models", **baseline_style)
        self.scatter_plots["baseline"] = scatter
        
        # Plot models to combine
        for i, (label, model) in enumerate(self.models_to_combine):
            metrics = self._evaluate_model(model)
            x_values.append(metrics[f'male_{self.metric}'])
            y_values.append(metrics[f'female_{self.metric}'])

            # Store metrics for interactivity
            model_id = f"tocombine_{i}"
            self.individual_models_data[model_id] = {
                'label': label,
                'type': 'tocombine',
                'metrics': metrics
            }
        
        scatter = self.ax.scatter(x_values, y_values, label="Models to Combine", **tocombine_style)
        self.scatter_plots["tocombine"] = scatter
        

    def _generate_zero_weight_combinations(self, n_combinations, zero_index):
        """Generate combinations where the specified model has zero weight"""
        num_models = len(self.models_to_combine)
        combinations = []
        
        for _ in range(n_combinations):
            # Generate weights for non-zero models
            non_zero_weights = np.random.dirichlet(np.ones(num_models - 1))
            
            # Insert zero at the specified index
            weights = np.insert(non_zero_weights, zero_index, 0)
            combinations.append(weights)
            
        return np.array(combinations)

    def generate_plot(self, n_combinations: int = 100):
        """Generate the main performance comparison plot"""
        num_models = len(self.models_to_combine)
        
        # Group 1: Standard combinations with all models
        standard_weights = np.random.dirichlet(np.ones(num_models), n_combinations)
        
        # Create a list to store all combination groups with their colors and labels
        self.combination_groups = [
            {"id": "all_models", "weights": standard_weights, "color": "blue", "label": "Combinations: All Models"}
        ]
        
        # Add zero-weight groups if we have 3 or more models
        if num_models >= 3:
            for i in range(min(3, num_models)):
                model_name = self.models_to_combine[i][0]
                zero_weights = self._generate_zero_weight_combinations(n_combinations // 3, i)
                
                self.combination_groups.append({
                    "id": f"without_{i}",
                    "weights": zero_weights,
                    "color": ["red", "green", "purple"][i],  # Different color for each group
                    "label": f"Without {model_name}"
                })
        
        # Evaluate all combinations
        self.metrics_data = []
        self.group_data = {}
        
        for group in self.combination_groups:
            group_metrics = []
            
            for w in tqdm(group["weights"], desc=f"Evaluating {group['label']}"):
                combined = self._combine_models(w)
                metrics = self._evaluate_model(combined)
                metrics.update({
                    'weights': w,
                    'group_id': group["id"],
                    'group_label': group["label"],
                    'color': group["color"]
                })
                group_metrics.append(metrics)
                self.metrics_data.append(metrics)
            
            self.group_data[group["id"]] = group_metrics

        # Create plot with adjusted figsize and more space for legend
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        
        # Adjust the subplot to make room for the legend
        plt.subplots_adjust(right=0.75)
        
        # Plot each group with its own color
        for group in self.combination_groups:
            group_id = group["id"]
            group_data = self.group_data[group_id]
            
            if group_data:
                x_values = [m[f'male_{self.metric}'] for m in group_data]
                y_values = [m[f'female_{self.metric}'] for m in group_data]
                
                scatter = self.ax.scatter(x_values, y_values, c=group["color"], 
                                        alpha=0.6, label=group["label"])
                self.scatter_plots[group_id] = scatter
        
        self._plot_basecomb_models()
        self._configure_plot()
        self._setup_interactivity()
        
        # Create and display the interactive dashboard
        display(self._create_display())

    def _configure_plot(self):
        """Configure plot aesthetics for InterpretML consistency"""
        self.ax.set_xlabel(f"Male {self.metric.title().replace('_', ' ')}",
                         fontsize=12)
        self.ax.set_ylabel(f"Female {self.metric.title().replace('_', ' ')}",
                         fontsize=12)
        
        if self.metric == "accuracy":
            self.ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
            
        self.ax.grid(True, alpha=0.3)
        
        # Organize legend better with too many entries
        handles, labels = self.ax.get_legend_handles_labels()
        
        # If we have more than 10 items, create a more compact legend
        if len(handles) > 10:
            # Group models by type and create custom legend entries
            by_type = {}
            # First add combination groups
            for group in self.combination_groups:
                by_type[group["label"]] = group["color"]
                
            # Then add baseline and model entries
            baseline_group = []
            model_group = []
            
            for model_id, data in self.individual_models_data.items():
                if data['type'] == 'baseline':
                    baseline_group.append(data['label'])
                else:
                    model_group.append(data['label'])
            
            # Use two-column layout for better space usage
            self.ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.02, 0.5),
                         frameon=False, fontsize=9, ncol=2)
        else:
            # Standard legend
            self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                         frameon=False, fontsize=10)
            
        # Set legend title
        legend = self.ax.get_legend()
        if legend:
            legend.set_title("Models", prop={'size': 12})

    def _setup_interactivity(self):
        """Add interactive tooltips with model details."""
        # Get all scatter plots (combinations and individual models)
        all_scatter_plots = list(self.scatter_plots.values())

        # Create cursor for all scatter plots
        cursor = mplcursors.cursor(all_scatter_plots)

        @cursor.connect("add")
        def on_add(sel):
            with self.info_output:
                clear_output(wait=True)

                # Find which scatter plot was selected
                selected_scatter = sel.artist
                point_index = sel.index

                # Check if it's the baseline scatter
                if selected_scatter == self.scatter_plots["baseline"]:
                    model_data = list(self.individual_models_data.values())[point_index]
                    display(widgets.HTML(
                        f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>"
                        f"<b>Baseline Model:</b> {model_data['label']}<br>"
                        f"<b>Male {self.metric.title()}:</b> {model_data['metrics'][f'male_{self.metric}']:.3f}<br>"
                        f"<b>Female {self.metric.title()}:</b> {model_data['metrics'][f'female_{self.metric}']:.3f}<br>"
                        f"<b>Overall {self.metric.title()}:</b> {model_data['metrics'][f'overall_{self.metric}']:.3f}"
                        "</div>"
                    ))
                else:
                    # Handle other scatter plots (e.g., combinations)
                    selected_id = None
                    for model_id, scatter in self.scatter_plots.items():
                        if scatter == selected_scatter:
                            selected_id = model_id
                            break

                    if selected_id and point_index < len(self.group_data.get(selected_id, [])):
                        metrics = self.group_data[selected_id][point_index]
                        weights_str = ', '.join([f"{name}: {w:.2f}" for name, w in zip(self.models_to_combine[:, 0], metrics['weights'])])
                        display(widgets.HTML(
                            f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>"
                            f"<b>Group:</b> {metrics['group_label']}<br>"
                            f"<b>Weights:</b> {weights_str}<br>"
                            f"<b>Male {self.metric.title()}:</b> {metrics[f'male_{self.metric}']:.3f}<br>"
                            f"<b>Female {self.metric.title()}:</b> {metrics[f'female_{self.metric}']:.3f}<br>"
                            f"<b>Overall {self.metric.title()}:</b> {metrics[f'overall_{self.metric}']:.3f}"
                            "</div>"
                        ))

    def _toggle_group_visibility(self, group_id, change):
        """Toggle visibility of a group's scatter plot"""
        if group_id in self.scatter_plots:
            scatter = self.scatter_plots[group_id]
            scatter.set_visible(change['new'])
            self.fig.canvas.draw_idle()

    def _create_checkboxes(self):
        """Create checkboxes for toggling group visibility"""
        checkbox_widgets = []
        
        # Add checkboxes for combination groups
        for group in self.combination_groups:
            group_id = group["id"]
            checkbox = widgets.Checkbox(
                value=True,
                description=group["label"],
                style={'description_width': 'initial'},
                layout=widgets.Layout(margin='5px 0')
            )
            checkbox.observe(lambda change, gid=group_id: self._toggle_group_visibility(gid, change), names='value')
            checkbox_widgets.append(checkbox)
        
        # Add a "Baseline Models" checkbox to toggle all baseline models
        if len(self.baseline_models) > 0:
            baseline_checkbox = widgets.Checkbox(
                value=True,
                description="Baseline Models",
                style={'description_width': 'initial'},
                layout=widgets.Layout(margin='5px 0')
            )
            baseline_checkbox.observe(lambda change: self._toggle_model_type_visibility('baseline', change), names='value')
            checkbox_widgets.append(baseline_checkbox)
        
        # Add a "Models to Combine" checkbox if needed
        if len(self.models_to_combine) > 0:
            combine_checkbox = widgets.Checkbox(
                value=True,
                description="Models to Combine",
                style={'description_width': 'initial'},
                layout=widgets.Layout(margin='5px 0')
            )
            combine_checkbox.observe(lambda change: self._toggle_model_type_visibility('combine', change), names='value')
            checkbox_widgets.append(combine_checkbox)
        
        return widgets.VBox(checkbox_widgets)
    
    def _toggle_model_type_visibility(self, model_type, change):
        """Toggle visibility for all models of a specific type"""
        for model_id, data in self.individual_models_data.items():
            if data['type'] == model_type:
                scatter = self.scatter_plots[model_id]
                scatter.set_visible(change['new'])
        self.fig.canvas.draw_idle()

    def _create_display(self):
        """Create final widget layout with checkboxes for visibility control"""
        # Create checkboxes for plot control
        checkboxes = self._create_checkboxes()
        
        # Create the control panel
        control_panel = widgets.VBox([
            widgets.HTML("<b>Model Details:</b>"),
            self.info_output,
            widgets.HTML("<b>Show/Hide Groups:</b>"),
            checkboxes
        ], layout={'width': '300px', 'margin': '0 20px'})
        
        # Make the figure canvas wider to accommodate the legend
        fig_canvas = self.fig.canvas
        fig_canvas.layout.width = '800px'
        
        return widgets.HBox([
            control_panel,
            fig_canvas
        ])