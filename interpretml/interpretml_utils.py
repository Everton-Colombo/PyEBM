import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, Output, Checkbox, HBox, VBox, Label, HTML
from IPython.display import display, clear_output
from typing import List, Union, Sequence, Literal
from interpret.glassbox._ebm._ebm import EBMModel, ExplainableBoostingClassifier, ExplainableBoostingRegressor
from interpret.glassbox import merge_ebms
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import mplcursors


class InterpretmlEBMVisualizer:
    def __init__(self, 
                 models: Union[Sequence['ExplainableBoostingClassifier'], Sequence['ExplainableBoostingRegressor']],
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


class CombinedEBM:
    
    def __init__(self, ebms: List[EBMModel], weights):
        self.ebms = ebms
        self.weights = np.array(weights) / sum(weights)  # Auto-normalize

        self.setting = "classification" if hasattr(ebms[0], "predict_proba") else "regression"
        
        self._setup_interface()
    
    def _setup_interface(self):
        if self.setting == "classification":
            self.predict_proba = self.__predict_proba
            self.predict = self.__predict_classification
        else:
            self.predict = self.__predict_regression
    
    def __predict_proba(self, X):
        probas = [ebm.predict_proba(X) * w 
                for ebm, w in zip(self.ebms, self.weights)]
        return np.sum(probas, axis=0)
    
    def __predict_classification(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def __predict_regression(self, X):
        return np.sum([ebm.predict(X) * w for ebm, w in zip(self.ebms, self.weights)], axis=0)
    
    def get_model_object(self):
        """Return a model object for InterpretML compatibility"""
        for ebm, w in zip(self.ebms, self.weights):
            ebm.term_scores_ = [[score * w for score in term_scores] for term_scores in ebm.term_scores_]
        
        return merge_ebms(self.ebms)


class GroupPerformanceAnalyzer:
    def __init__(self, male_model: ExplainableBoostingClassifier, 
                 female_model: ExplainableBoostingClassifier,
                 normal_model: ExplainableBoostingClassifier,
                 X_test: pd.DataFrame, y_test: np.ndarray,
                 X_train: pd.DataFrame = None, y_train: np.ndarray = None,
                 male_mask: np.ndarray = None, female_mask: np.ndarray = None,
                 feature_of_interest: str = 'sex',
                 combine_strategy: Literal["pre", "post"] = "post",
                 metric: Literal["accuracy", "log_likelihood", "auc"] = "accuracy"):
        
        self.male_model = male_model
        self.female_model = female_model
        self.normal_model = normal_model
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.feature_of_interest = feature_of_interest
        self.combine_strategy = combine_strategy
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
        self.scatter = []
        self.info_output = Output()
        self.metrics_data = []
    
    def _get_weighed_model(self, model: EBMModel, weight: float) -> EBMModel:
        new_model = model
        
        if hasattr(new_model, 'predict_proba'):
            new_model.predict_proba = lambda X: model.predict_proba(X) * weight
        else:
            new_model.predict = lambda X: model.predict(X) * weight
        
        return new_model

    def _combine_models(self, male_weight: float, female_weight: float) -> ExplainableBoostingClassifier:
        """Combine models using InterpretML's API capabilities"""
        if self.combine_strategy == "post":
            # male_model = self._get_weighed_model(self.male_model, male_weight)
            # female_model = self._get_weighed_model(self.female_model, female_weight)
            # return merge_ebms([male_model, female_model])
            return CombinedEBM([self.male_model, self.female_model], [male_weight, female_weight])
        elif self.combine_strategy == "pre":
            # Pre-combination strategy - train new model with sample weights
            sample_weights = (
                male_weight * self.male_mask.astype(float) +
                female_weight * self.female_mask.astype(float)
            )
            
            # Clone and retrain model with InterpretML's EBM
            model = ExplainableBoostingClassifier()
            model.fit(self.X_train, self.y_train, sample_weight=sample_weights)
            
        return model

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

    def _setup_base_models(self):
        """Initialize baseline models with InterpretML-specific handling"""
        base_models = [
            ('Male Model', self.male_model),
            ('Female Model', self.female_model),
            ('Normal Model', self.normal_model),
            ('50-50 Model', self._create_50_50_model())
        ]
        
        for label, model in base_models:
            metrics = self._evaluate_model(model)
            self.scatter.append(self._plot_baseline_model(metrics, label))

    def _create_50_50_model(self):
        """Create 50-50 averaged model using InterpretML's predict_proba"""
        return self._combine_models(0.5, 0.5)

    def _plot_baseline_model(self, metrics: dict, label: str):
        """Plot model metrics with InterpretML-style formatting"""
        x_val = metrics[f'male_{self.metric}']
        y_val = metrics[f'female_{self.metric}']
        
        if label == '50-50 Model':
            return self.ax.scatter(x_val, y_val, s=200, marker='*', 
                          c='gold', edgecolors='black', 
                          label=label, zorder=5)
        else:
            return self.ax.scatter(x_val, y_val, s=100, edgecolors='black', 
                          label=label, zorder=10)

        self.metrics_data.append(metrics)

    def generate_plot(self, n_combinations: int = 100):
        """Generate the main performance comparison plot"""
        weights = np.random.dirichlet(np.ones(2), n_combinations)
        
        # Evaluate combinations
        for mw, fw in tqdm(weights, desc="Evaluating combinations"):
            combined = self._combine_models(mw, fw)
            metrics = self._evaluate_model(combined)
            metrics.update({'male_weight': mw, 'female_weight': fw})
            self.metrics_data.append(metrics)

        # Create plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        x_values = [m[f'male_{self.metric}'] for m in self.metrics_data]
        y_values = [m[f'female_{self.metric}'] for m in self.metrics_data]
        
        self.scatter.append(self.ax.scatter(x_values, y_values, c='blue', alpha=0.6))
        self._setup_base_models()
        self._configure_plot()
        self._setup_interactivity()
        
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
        self.ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                     borderaxespad=0., frameon=False)

    def _setup_interactivity(self):
        """Add interactive tooltips with model details"""
        cursor = mplcursors.cursor(self.scatter)
        
        @cursor.connect("add")
        def on_add(sel):
            with self.info_output:
                clear_output(wait=True)
                metrics = self.metrics_data[sel.index]
                display(HTML(
                    f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>"
                    f"<b>Male Weight:</b> {metrics.get('male_weight', 'N/A'):.2f}<br>"
                    f"<b>Female Weight:</b> {metrics.get('female_weight', 'N/A'):.2f}<br>"
                    f"<b>Male {self.metric.title()}:</b> {metrics[f'male_{self.metric}']:.3f}<br>"
                    f"<b>Female {self.metric.title()}:</b> {metrics[f'female_{self.metric}']:.3f}<br>"
                    f"<b>Overall {self.metric.title()}:</b> {metrics[f'overall_{self.metric}']:.3f}"
                    "</div>"
                ))

    def _create_display(self):
        """Create final widget layout"""
        return HBox([
            VBox([HTML("<b>Model Details:</b>"),
                  self.info_output],
                layout={'width': '300px', 'margin': '0 20px'}),
            self.fig.canvas
        ])