def test_model(model, x_test, y_test):

    score = model.evaluate(x_test, y_test)

    return score
