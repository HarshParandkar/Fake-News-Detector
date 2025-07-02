# test_runner.py

from models.logistic_regression_model import train_logistic_regression
from models.random_forest_model import train_random_forest
from models.naive_bayes_model import train_naive_bayes
from models.decision_tree_model import train_decision_tree
from models.xgboost_model import train_xgboost
from models.svm_model import train_svm

def main():
    # print("Testing Logistic Regression:")
    # train_logistic_regression()

    # print("\nTesting Random Forest:")
    # train_random_forest()

    # print("\nTesting Naive Bayes:")
    # train_naive_bayes()

    # print("\nTesting Decision Tree:")
    # train_decision_tree()

    print("\nTesting XGBoost:")
    train_xgboost()

    # print("\nTesting SVM:")
    # train_svm()

if __name__ == "__main__":
    main()
