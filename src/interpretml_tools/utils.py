from typing import List, Dict, Any
import numpy as np
from interpret.glassbox._ebm._ebm import EBMModel, ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import accuracy_score, r2_score


class CombinedEBM(BaseEstimator):
    """
    This class combines multiple EBM models into a single model.
    Contrary to interpretML's merge_ebms function, this class performs very few checks,
    and assumes that the models are compatible.
    
    The combination is done by a weighted average of the predictions from each model.
    The feature functions are also combined in a similar way.
    
    This class implements the scikit-learn estimator interface.
    """
    
    def __init__(self, ebms: List[EBMModel], weights=None):
        self.ebms = ebms
        self.weights = weights
        
    def fit(self, X=None, y=None):
        """
        This method is implemented for scikit-learn compatibility.
        Since this class combines pre-trained models, it doesn't need actual training.
        
        Returns
        -------
        self : object
            Returns self.
        """
        weights = self.weights
        if weights is None:
            weights = [1.0] * len(self.ebms)
            
        self.weights_ = np.array(weights) / sum(weights)  # Auto-normalize

        self.setting_ = "classification" if hasattr(self.ebms[0], "predict_proba") else "regression"
        
        # Copy attributes from the first model
        self.feature_names_in_ = self.ebms[0].feature_names_in_
        self.term_features_ = self.ebms[0].term_features_
        self.intercept_ = sum(ebm.intercept_ * w for ebm, w in zip(self.ebms, self.weights_))
        
        # Combine bins, term_scores, and feature_bounds
        self._combine_model_structures()
        
        self._setup_interface()
        return self
    
    def _setup_interface(self):
        if self.setting_ == "classification":
            self.classes_ = self.ebms[0].classes_
            self.n_classes_ = len(self.classes_)
            self.predict_proba = self.__predict_proba
            self.predict = self.__predict_classification
        else:
            self.predict = self.__predict_regression
    
    def __predict_proba(self, X):
        probas = [ebm.predict_proba(X) * w 
                for ebm, w in zip(self.ebms, self.weights_)]
        return np.sum(probas, axis=0)
    
    def __predict_classification(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def __predict_regression(self, X):
        return np.sum([ebm.predict(X) * w for ebm, w in zip(self.ebms, self.weights_)], axis=0)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels for classification,
        or the coefficient of determination R^2 for regression.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for X.
            
        Returns
        -------
        score : float
            Mean accuracy or R^2 score
        """
        if self.setting_ == "classification":
            return accuracy_score(y, self.predict(X))
        else:
            return r2_score(y, self.predict(X))
            
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        params = {
            "ebms": self.ebms,
            "weights": self.weights
        }
        return params
        
    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters.
            
        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
    
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
        for ebm, weight in zip(self.ebms, self.weights_):
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
        for ebm, weight in zip(self.ebms, self.weights_):
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
