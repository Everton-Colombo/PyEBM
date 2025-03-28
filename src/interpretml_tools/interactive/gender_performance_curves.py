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
    
    def _setup_combination_models(self, n_combinations):
        """Creates the combination groups, combines the models and stores the results"""
        
        ### Setting up combination groups ###
        combination_group_model_indexes: list[list[int]] = [] # list of list of ints containing indexes of models in each group
        combination_group_model_indexes.append(list(range(len(self.models_to_combine)))) # All models group
        
        # Add models three by three (if possible)
        if len(self.models_to_combine) >= 3:
            for i in range(0, len(self.models_to_combine) - 1):
                combination_group_model_indexes.append(list(range(i, i + 2)))
        
        # Process each group
        for i, group in enumerate(combination_group_model_indexes):
            # Generate weights for the current group
            weight_sets = np.random.dirichlet(np.ones(len(group)), n_combinations)
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
        
        
    def generate_plot(self, n_combinations: int = 100):
        self._setup_combination_models(n_combinations)
        self._setup_individual_models()
        
        self._plot_model_groups()
        self._configure_plot()
        self._setup_interactivity()
        
        display(self._create_display())
    
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
            
    
    def _create_display(self):
        """Create final widget layout with checkboxes for visibility control"""
        # Create checkboxes for plot control
        checkboxes = self._create_togglevis_checkboxes()
        
        # Create the control panel
        control_panel = widgets.VBox([
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