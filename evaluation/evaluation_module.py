import numpy as np


def mean_iou(y_test, y_pred, n_classes):

    """
    Calculates mean intersection over union between test
    data and predictions.
    """

    ious = []
    for i in range(y_test.shape[0]):

        y_test_i = np.squeeze(y_test[i])
        y_pred_i = y_pred[i]

        for c in range(n_classes):
            epsilon = 1e-5
            act = np.where(y_test_i == c)[0]
            pred = np.where(y_pred_i == c)[0]

            i = np.intersect1d(act, pred).shape[0]
            u = np.union1d(act, pred).shape[0]
            iou = (i + epsilon) / (u + epsilon)
            ious.append(iou)

    iou = np.mean(ious)
    print(f"Mean IoU Score: {iou}")


def evaluate_sample(model, X_test, y_test, idx):

    #     decode X
    X_i = (X_test[idx].reshape((128)) * (126 - 32)) + 32
    X_i = np.array([chr(int(i)) for i in X_i])
    print("X_sample:")
    print("")
    print(X_i)

    #     decode y
    y_pred = model.predict(X_test[idx].reshape(1, 128, 1))
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = y_pred.reshape((128))

    #     organize prediction elements
    number_pred = "".join(X_i[np.where(y_pred == 1)[0]])
    street_pred = "".join(X_i[np.where(y_pred == 2)[0]])
    city_pred = "".join(X_i[np.where(y_pred == 3)[0]])
    postcode_pred = "".join(X_i[np.where(y_pred == 4)[0]])
    country_pred = "".join(X_i[np.where(y_pred == 5)[0]])

    y_act = y_test[idx].reshape((128))
    number_act = "".join(X_i[np.where(y_act == 1)[0]])
    street_act = "".join(X_i[np.where(y_act == 2)[0]])
    city_act = "".join(X_i[np.where(y_act == 3)[0]])
    postcode_act = "".join(X_i[np.where(y_act == 4)[0]])
    country_act = "".join(X_i[np.where(y_act == 5)[0]])

    d = {}
    d["number"] = (number_pred, number_act)
    d["street"] = (street_pred, street_act)
    d["city"] = (city_pred, city_act)
    d["postcode"] = (postcode_pred, postcode_act)
    d["country"] = (country_pred, country_act)

    print("")
    print("{:>30}{:>27}".format("Predicted:", "Actual:"))
    print("")
    for k, v in d.items():
        print(f"{k:<20}{v[0]:<30}{v[1]}")
