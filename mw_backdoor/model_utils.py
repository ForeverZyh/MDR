import lightgbm as lgb
from mw_backdoor.transfer_3 import Deep_NN
def train_model(model_id, x_train, y_train):
    """ Train an EmberNN classifier

    :param model_id: (str) model type
    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained classifier
    """

    if model_id == 'lightgbm':
        return train_lightgbm(
            x_train=x_train,
            y_train=y_train
        )
    elif model_id == "DNN":
        return train_DNN(
            x_train=x_train,
            y_train=y_train
        )

    else:
        raise NotImplementedError('Model {} not supported'.format(model_id))

def train_lightgbm(x_train, y_train):
    """ Train a LightGBM classifier

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained LightGBM classifier
    """

    lgbm_dataset = lgb.Dataset(x_train, y_train)
    lgbm_model = lgb.train({"application": "binary",'verbose':-1}, lgbm_dataset)

    return lgbm_model

def train_DNN(x_train, y_train):
    """ Train a DNN classifier

    :param x_train: (ndarray) train data
    :param y_train: (ndarray) train labels
    :return: trained DNN classifier
    """

    model = Deep_NN(2351)
    model.fit(x_train, y_train)
    return model