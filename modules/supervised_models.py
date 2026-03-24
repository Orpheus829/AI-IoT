from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

class FailurePredictionBaseline:
    """Classical ML models for failure prediction"""
    
    def __init__(self):
        self.models = {
        'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=5, 
                random_state=42
            ),
            'SVM': SVC(
                kernel='rbf', 
                probability=True, 
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42
            ),
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(64, 32), 
                max_iter=500, 
                random_state=42
            )
        }
        
        self.best_model = None
        self.best_score = 0
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train all models and compare"""
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()
            
            # Test accuracy
            test_score = model.score(X_test, y_test)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            results[name] = {
                'model': model,
                'cv_score': cv_score,
                'test_score': test_score,
                'predictions': y_pred
            }
            
            print(f"  CV Score: {cv_score:.4f}")
            print(f"  Test Score: {test_score:.4f}")
            
            # Track best model
            if cv_score > self.best_score:
                self.best_score = cv_score
                self.best_model = (name, model)
        
        return results
    
    def get_best_model(self):
        """Return best performing model"""
        return self.best_model