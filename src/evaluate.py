
import matplotlib.pyplot as plt          # For plotting graphs and figures
import seaborn as sns                    # For better looking plots (heatmaps, etc.)
import numpy as np                       # For numerical operations
import pandas as pd                      # For data handling
import os                                # For file and folder operations
import joblib                            # (Optional) For saving/loading models
from sklearn.metrics import (            # Evaluation metrics from scikit-learn
    classification_report,               # Precision, Recall, F1-score table
    roc_auc_score,                       # ROC-AUC metric
    confusion_matrix,                    # Confusion matrix
    roc_curve,                           # ROC curve values
    precision_recall_curve               # Precision-Recall curve values
)

# -------------------------------------------------
# Function: Evaluate a single model
# -------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name="Model", save_path="reports/figures/"):
    """
    This function evaluates a trained machine learning model on test data.
    It prints metrics, plots graphs, saves figures, and returns key scores.

    Args:
        model: Trained ML model (e.g., Logistic Regression, RF, XGBoost)
        X_test: Test features
        y_test: True labels for test data
        model_name: Name of the model (for plot titles and filenames)
        save_path: Folder where figures will be saved
    """

    # Create the folder if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # -------------------------------------------------
    # Generate predictions
    # -------------------------------------------------
    # Predicted class labels (0 or 1)
    y_pred = model.predict(X_test)

    # Predicted probabilities for the positive class (diabetes = 1)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # -------------------------------------------------
    # Print classification report and ROC-AUC score
    # -------------------------------------------------
    print(f"\n{model_name} Performance:")
    print(classification_report(y_test, y_pred))  # Shows precision, recall, F1-score per class
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")  # Overall ROC-AUC score
    
    # -------------------------------------------------
    # Confusion Matrix Plot
    # -------------------------------------------------
    plt.figure(figsize=(8, 6))

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix as a heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Non-diabetic', 'Diabetic'],
        yticklabels=['Non-diabetic', 'Diabetic']
    )

    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()

    # Save the confusion matrix figure
    plt.savefig(f"{save_path}/{model_name.lower()}_confusion_matrix.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------
    # ROC Curve Plot
    # -------------------------------------------------
    plt.figure(figsize=(10, 8))

    # Get False Positive Rate and True Positive Rate
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    # Plot ROC curve
    plt.plot(
        fpr, tpr,
        label=f'{model_name} (AUC = {roc_auc_score(y_test, y_proba):.2f})',
        linewidth=2
    )

    # Plot diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    # Save ROC curve figure
    plt.savefig(f"{save_path}/{model_name.lower()}_roc_curve.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------
    # Precision-Recall Curve Plot
    # -------------------------------------------------
    plt.figure(figsize=(10, 6))

    # Compute precision and recall values for different thresholds
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    # Plot Precision-Recall curve
    plt.plot(recall, precision, label=model_name, linewidth=2)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    # Save Precision-Recall curve figure
    plt.savefig(f"{save_path}/{model_name.lower()}_pr_curve.png", dpi=300)
    plt.close()
    
    # -------------------------------------------------
    # Feature Importance Plot (only for tree-based models)
    # -------------------------------------------------
    # Some models like Random Forest / XGBoost have feature_importances_
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))

        # Create a Pandas Series with feature names and importance scores
        importance = pd.Series(model.feature_importances_, index=X_test.columns)

        # Sort and plot as horizontal bar chart
        importance.sort_values().plot(kind='barh', color='skyblue')

        plt.title(f'{model_name} Feature Importance')
        plt.xlabel('Importance Score')

        # Save feature importance plot
        plt.savefig(f"{save_path}/{model_name.lower()}_feature_importance.png", dpi=300)
        plt.close()
    
    # -------------------------------------------------
    # Return key results for later comparison
    # -------------------------------------------------
    return {
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'roc_auc': roc_auc_score(y_test, y_proba)
    }

# -------------------------------------------------
# Function: Compare multiple models
# -------------------------------------------------
def compare_models(results_dict, save_path="reports/figures/"):
    """
    This function compares multiple models using key metrics
    and plots a single comparison graph.

    Args:
        results_dict: Dictionary containing results of multiple models
                      (output of evaluate_model for each model)
        save_path: Folder to save the comparison plot
    """

    # Metrics to compare
    metrics = ['precision', 'recall', 'f1-score', 'roc_auc']
    comparison = {}
    
    # Extract metrics from each model's results
    for model_name, results in results_dict.items():
        comparison[model_name] = {
            'precision': results['classification_report']['weighted avg']['precision'],
            'recall': results['classification_report']['weighted avg']['recall'],
            'f1-score': results['classification_report']['weighted avg']['f1-score'],
            'roc_auc': results['roc_auc']
        }
    
    # -------------------------------------------------
    # Plot comparison graph
    # -------------------------------------------------
    plt.figure(figsize=(14, 8))

    # Plot each metric across all models
    for metric in metrics:
        values = [comparison[model][metric] for model in comparison]
        plt.plot(list(comparison.keys()), values, 'o-', label=metric)
    
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0.5, 1.0)      # Limit y-axis for better visualization
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=15)
    plt.tight_layout()

    # Save comparison plot
    plt.savefig(f"{save_path}/model_comparison.png", dpi=300)
    plt.close()

# -------------------------------------------------
# Main block (runs only if this file is executed directly)
# -------------------------------------------------
if __name__ == "__main__":
    # Example usage: train a model and evaluate it
    from train import train_model
    
    # Train the model and get test data
    model, X_test, y_test = train_model('data/diabetes.csv')

    # Evaluate the trained model
    results = evaluate_model(model, X_test, y_test)