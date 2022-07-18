from preprocessing.data_pipeline import run_data_pipeline
from model.transfer_learning_model import compile_model
from common.constants import ModelConstants

if __name__ == "__main__":
    training_generator, validation_generator = run_data_pipeline(split=False)
    print("Data ready.")
    model = compile_model()
    print(model.summary())
    print("****** BEGIN TRAINING ******")
    train_hist = model.fit(
        training_generator,
        epochs=ModelConstants.EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    print("****** END TRAINING ******")
