import os
from datetime import datetime

import mlflow
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


class SVMLogger:

    def __init__(self, log_path=None):
        self.log_path = log_path

    def before_experiment(self, train_site_name, test_site_name, experiment_description):
        pass

    def after_experiment(self):
        pass

    def before_train_and_test(self, c, kernel):
        pass

    def after_train_and_test(self, c, kernel, models, metrics):
        pass

    def before_train(self, train_site_name, test_site_name, train_dir, test_dir, test_ratio, train_labels_mean, c, kernel, experiment_description):
        pass

    def before_test(self, test_labels_mean):
        pass

    def make_confusion_matrix_image(self, cm, model):
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        display.plot()



class MLFlowLogger(SVMLogger):
    def before_experiment(self, train_site_name, test_site_name, experiment_description):
        experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mlflow.set_experiment(
            f"{experiment_time};Train:{train_site_name},test:{test_site_name},extra:{experiment_description}"
        )

    def after_experiment(self):
        mlflow.end_run()

    def before_train_and_test(self, c, kernel):
        mlflow.start_run()
        experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_name = f"{experiment_time};c={c},ker={kernel}"
        mlflow.set_tag("mlflow.runName", experiment_name)
        experiment_path = os.path.join(self.log_path, experiment_name)
        os.mkdir(path=experiment_path)
        self.current_experiment_path = experiment_path

    def after_train_and_test(self, c, kernel, models, metrics):
        experiment_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        experiment_name = f"{experiment_time};c={c},ker={kernel}"
        mlflow.sklearn.log_model(models[-1], experiment_name)
        mlflow.log_metrics(metrics[-1], synchronous=True)

    def before_train(self, train_site_name, test_site_name, train_dir, test_dir, test_ratio, train_labels_mean, c, kernel, experiment_description):
        mlflow_tracking_params = {
            'Train site': train_site_name,
            'Train directory': train_dir,
            'Test site': test_site_name,
            'Test directory': test_dir,
            'Test_ratio': test_ratio,
            'Train_labels_mean': train_labels_mean,
            'C': c,
            'Kernel': kernel,
            'extra info': experiment_description
        }
        mlflow.log_params(params=mlflow_tracking_params, synchronous=True)

    def before_test(self, test_labels_mean):
        mlflow.log_param(key='Test_labels_mean', value=test_labels_mean)

    def make_confusion_matrix_image(self, cm, model):
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        display.plot().figure_.savefig(os.path.join(self.current_experiment_path, 'confusion_matrix.png'))
        plt.close('all')
        mlflow.log_artifact(os.path.join(self.current_experiment_path, 'confusion_matrix.png'))