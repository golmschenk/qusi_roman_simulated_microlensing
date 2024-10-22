from qusi.model import Hadryss
from qusi.session import TrainHyperparameterConfiguration, train_session

from qusi_roman_simulated_microlensing.dataset import get_train_dataset, get_validation_dataset


def main():
    train_light_curve_dataset = get_train_dataset()
    validation_light_curve_dataset = get_validation_dataset()
    model = Hadryss.new(input_length=42227)
    train_hyperparameter_configuration = TrainHyperparameterConfiguration.new(
        batch_size=100, cycles=200, train_steps_per_cycle=500, validation_steps_per_cycle=100)
    train_session(train_datasets=[train_light_curve_dataset], validation_datasets=[validation_light_curve_dataset],
                  model=model, hyperparameter_configuration=train_hyperparameter_configuration)


if __name__ == '__main__':
    main()
