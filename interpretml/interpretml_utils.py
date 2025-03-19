import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, Dropdown, Output, Checkbox, HBox, VBox, Label, HTML
import ipywidgets as widgets
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
        
        # Copy attributes from the first model
        self.feature_names_in_ = ebms[0].feature_names_in_
        self.term_features_ = ebms[0].term_features_
        self.intercept_ = sum(ebm.intercept_ * w for ebm, w in zip(ebms, self.weights))
        
        # Combine bins, term_scores, and feature_bounds
        self._combine_model_structures()
        
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
    
    def _combine_model_structures(self):
        """Combine the internal structures of the component EBMs."""
        self.bins_ = []
        self.term_scores_ = []
        self.feature_bounds_ = []
        
        # Process each feature
        for feature_idx in range(len(self.feature_names_in_)):
            # Find the corresponding term for this feature in the first model
            term_idx = None
            for i, term in enumerate(self.term_features_):
                if len(term) == 1 and term[0] == feature_idx:
                    term_idx = i
                    break
            
            if term_idx is None:
                # Feature not used in model, create a placeholder
                self.bins_.append(self.ebms[0].bins_[feature_idx])
                self.term_scores_.append(np.zeros_like(self.ebms[0].term_scores_[term_idx]))
                self.feature_bounds_.append(self.ebms[0].feature_bounds_[feature_idx])
                continue
            
            # Check if this is a categorical or continuous feature
            first_bin = self.ebms[0].bins_[feature_idx]
            if isinstance(first_bin[0], dict):
                # Categorical feature
                self._combine_categorical_feature(feature_idx, term_idx)
            else:
                # Continuous feature
                self._combine_continuous_feature(feature_idx, term_idx)
    
    def _combine_categorical_feature(self, feature_idx, term_idx):
        """Combine a categorical feature across models."""
        # Get all unique categories across models
        all_categories = set()
        for ebm in self.ebms:
            bin_info = ebm.bins_[feature_idx]
            all_categories.update(bin_info[0].keys())
        
        # Create combined bin information
        combined_bin_dict = {}
        for category in all_categories:
            combined_bin_dict[category] = len(combined_bin_dict)
        
        # Create combined scores
        combined_scores = np.zeros(len(combined_bin_dict) + 2)  # +2 for missing and unseen bins
        
        # Add weighted scores from each model
        for ebm, weight in zip(self.ebms, self.weights):
            bin_info = ebm.bins_[feature_idx]
            categories_dict = bin_info[0]
            
            # Find the term index in this model
            model_term_idx = None
            for i, term in enumerate(ebm.term_features_):
                if len(term) == 1 and term[0] == feature_idx:
                    model_term_idx = i
                    break
            
            if model_term_idx is not None:
                scores = ebm.term_scores_[model_term_idx]
                
                # Transfer missing bin score
                combined_scores[0] += scores[0] * weight
                
                # Transfer scores for each category
                for category, idx in categories_dict.items():
                    combined_idx = combined_bin_dict[category] + 1  # +1 to account for missing bin
                    combined_scores[combined_idx] += scores[idx + 1] * weight  # +1 in scores to account for missing bin
                
                # Transfer unseen bin score
                combined_scores[-1] += scores[-1] * weight
        
        self.bins_.append([combined_bin_dict])
        
        # Create or update the term for this feature
        if term_idx < len(self.term_scores_):
            self.term_scores_[term_idx] = combined_scores
        else:
            self.term_scores_.append(combined_scores)
        
        # Use feature bounds from first model
        self.feature_bounds_.append(self.ebms[0].feature_bounds_[feature_idx])
    
    def _combine_continuous_feature(self, feature_idx, term_idx):
        """Combine a continuous feature across models using weighted interpolation."""
        # Get the combined set of bin edges
        all_bin_edges = set()
        for ebm in self.ebms:
            bin_info = ebm.bins_[feature_idx]
            cuts = bin_info[0]
            all_bin_edges.update(cuts)
        
        # Add feature min and max bounds
        feature_min = min(ebm.feature_bounds_[feature_idx][0] for ebm in self.ebms)
        feature_max = max(ebm.feature_bounds_[feature_idx][1] for ebm in self.ebms)
        
        # Sort the bin edges
        all_bin_edges = sorted(all_bin_edges)
        
        # Create combined bin edges
        combined_cuts = np.array(all_bin_edges)
        
        # Create interpolated scores for the combined bins
        combined_scores = np.zeros(len(combined_cuts) + 3)  # +3 for missing, leftmost, and rightmost bins
        
        # Add weighted scores from each model
        for ebm, weight in zip(self.ebms, self.weights):
            bin_info = ebm.bins_[feature_idx]
            cuts = bin_info[0]
            
            # Find the term index in this model
            model_term_idx = None
            for i, term in enumerate(ebm.term_features_):
                if len(term) == 1 and term[0] == feature_idx:
                    model_term_idx = i
                    break
            
            if model_term_idx is not None:
                scores = ebm.term_scores_[model_term_idx]
                
                # Transfer missing bin score
                combined_scores[0] += scores[0] * weight
                
                # Interpolate scores for the main bins
                model_bin_edges = np.concatenate([[feature_min], cuts, [feature_max]])
                bin_midpoints = 0.5 * (model_bin_edges[:-1] + model_bin_edges[1:])
                
                # Create a function to interpolate between bin midpoints and scores
                from scipy.interpolate import interp1d
                interp_func = interp1d(
                    bin_midpoints, 
                    scores[1:-1],  # Skip missing and unseen bins
                    bounds_error=False,
                    fill_value=(scores[1], scores[-2])  # Use leftmost and rightmost bin values for out-of-bounds
                )
                
                # Compute midpoints for combined bins
                combined_bin_edges = np.concatenate([[feature_min], combined_cuts, [feature_max]])
                combined_midpoints = 0.5 * (combined_bin_edges[:-1] + combined_bin_edges[1:])
                
                # Interpolate scores for each bin in the combined model
                interpolated_scores = interp_func(combined_midpoints)
                
                # Add weighted scores to the combined scores
                combined_scores[1:-1] += interpolated_scores * weight
                
                # Transfer unseen bin score
                combined_scores[-1] += scores[-1] * weight
        
        self.bins_.append([combined_cuts])
        
        # Create or update the term for this feature
        if term_idx < len(self.term_scores_):
            self.term_scores_[term_idx] = combined_scores
        else:
            self.term_scores_.append(combined_scores)
        
        # Use combined feature bounds
        self.feature_bounds_.append([feature_min, feature_max])
    
    def get_model_object(self):
        """Return a model object for InterpretML compatibility"""
        return self  # The enhanced CombinedEBM class is now directly compatible

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
        
        self.metrics_data.append(metrics)
        
        if label == '50-50 Model':
            return self.ax.scatter(x_val, y_val, s=200, marker='*', 
                          c='gold', edgecolors='black', 
                          label=label, zorder=5)
        else:
            return self.ax.scatter(x_val, y_val, s=100, edgecolors='black', 
                          label=label, zorder=10)

        

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
                    f"<b>Index:</b> {sel.index}<br>"
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

    def _plot_baseline_models(self):
        """Initialize baseline models with InterpretML-specific handling"""
        base_models = self.baseline_models.tolist() + self.models_to_combine.tolist()
        
        for label, model in base_models:
            metrics = self._evaluate_model(model)
            x_val = metrics[f'male_{self.metric}']
            y_val = metrics[f'female_{self.metric}']
            
            self.ax.scatter(x_val, y_val, s=100, edgecolors='black', 
                            label=label, zorder=10)   

    def generate_plot(self, n_combinations: int = 100):
        """Generate the main performance comparison plot"""
        weights = np.random.dirichlet(np.ones(len(self.models_to_combine)), n_combinations)
        
        # Evaluate combinations
        for w in tqdm(weights, desc="Evaluating combinations"):
            combined = self._combine_models(w)
            metrics = self._evaluate_model(combined)
            metrics.update({'weights': w})
            self.metrics_data.append(metrics)

        # Create plot
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        x_values = [m[f'male_{self.metric}'] for m in self.metrics_data]
        y_values = [m[f'female_{self.metric}'] for m in self.metrics_data]
        
        self.scatter.append(self.ax.scatter(x_values, y_values, c='blue', alpha=0.6))
        self._plot_baseline_models()
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
                weights_str = ', '.join([f"{name}: {w:.2f}" for name, w in zip(self.models_to_combine[:, 0], metrics['weights'])])
                display(HTML(
                    f"<div style='border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>"
                    f"<b>Weights:</b> {weights_str}<br>"
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
        self.info_output = Output()
        self.metrics_data = []
        self.combination_groups = []
        self.group_data = {}  # Dictionary to store data by group
    
    def _get_weighed_model(self, model: EBMModel, weight: float) -> EBMModel:
        new_model = model
        
        if hasattr(new_model, 'predict_proba'):
            new_model.predict_proba = lambda X: model.predict_proba(X) * weight
        else:
            new_model.predict = lambda X: model.predict(X) * weight
        
        return new_model

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

    def _plot_baseline_models(self):
        """Initialize baseline models with InterpretML-specific handling"""
        base_models = self.baseline_models.tolist() + self.models_to_combine.tolist()
        
        for label, model in base_models:
            metrics = self._evaluate_model(model)
            x_val = metrics[f'male_{self.metric}']
            y_val = metrics[f'female_{self.metric}']
            
            self.ax.scatter(x_val, y_val, s=100, edgecolors='black', 
                            label=label, zorder=10)   

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
            {"id": "all_models", "weights": standard_weights, "color": "blue", "label": "All Models"}
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
        
        self._plot_baseline_models()
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
        
        # Move legend outside the plot to prevent cutting
        legend = self.ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                             frameon=False, fontsize=10)
        legend.set_title("Groups", prop={'size': 12})

    def _setup_interactivity(self):
        """Add interactive tooltips with model details"""
        cursor = mplcursors.cursor(list(self.scatter_plots.values()))
        
        @cursor.connect("add")
        def on_add(sel):
            with self.info_output:
                clear_output(wait=True)
                
                # Find which scatter plot was selected
                selected_scatter = sel.artist
                point_index = sel.index
                
                # Find the group that this scatter plot belongs to
                selected_group_id = None
                for group_id, scatter in self.scatter_plots.items():
                    if scatter == selected_scatter:
                        selected_group_id = group_id
                        break
                
                if selected_group_id and point_index < len(self.group_data[selected_group_id]):
                    metrics = self.group_data[selected_group_id][point_index]
                    
                    weights_str = ', '.join([f"{name}: {w:.2f}" for name, w in zip(self.models_to_combine[:, 0], metrics['weights'])])
                    display(HTML(
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
        
        for group in self.combination_groups:
            group_id = group["id"]
            checkbox = Checkbox(
                value=True,
                description=group["label"],
                style={'description_width': 'initial'},
                layout=
                widgets.Layout(margin='5px 0')
            )
            checkbox.observe(lambda change, gid=group_id: self._toggle_group_visibility(gid, change), names='value')
            checkbox_widgets.append(checkbox)
        
        return VBox(checkbox_widgets)

    def _create_display(self):
        """Create final widget layout with checkboxes for visibility control"""
        # Create checkboxes for plot control
        checkboxes = self._create_checkboxes()
        
        # Create the control panel
        control_panel = VBox([
            HTML("<b>Model Details:</b>"),
            self.info_output,
            HTML("<b>Show/Hide Groups:</b>"),
            checkboxes
        ], layout={'width': '300px', 'margin': '0 20px'})
        
        # Make the figure canvas wider to accommodate the legend
        fig_canvas = self.fig.canvas
        fig_canvas.layout.width = '800px'
        
        return HBox([
            control_panel,
            fig_canvas
        ])