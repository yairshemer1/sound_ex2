from genre_classifier import *


def test_train(handler, training_params):
    try:
        handler.train_new_model(training_params)
        print("Train dummy test passed")
    except Exception as e:
        print(f"Train dummy test failed, exception:\n{e}")


def test_pre_trained(handler, training_params):
    try:
        pretrained_model = handler.get_pretrained_model()
        print("Get pretrained object dummy test passed")
    except Exception as e:
        print(f"Get pretrained object dummy test failed, exception:\n{e}")
    test_dataset = DataSet(json_dir=training_params.test_json_path)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
    test_wavs, test_labels = next(test_loader.__iter__())
    test_pred = pretrained_model.classify(test_wavs.unsqueeze(1))
    evaluate_model(test_labels, test_pred.squeeze(), 'dummy_test')


if __name__ == "__main__":
    handler = ClassifierHandler()
    training_params = TrainingParameters(batch_size=1, num_epochs=1, save_dir="dummy_test")

    test_train(handler, training_params)

    # check that model object is obtained
    test_pre_trained(handler, training_params)

    # feel free to add tests here.
    # We will not be giving score to submitted tests.
    # You may (and recommended to) share tests with one another.
