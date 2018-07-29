

def load_model(
    model_name,
    epochs,
    batch_size,
    training,
    img_dim=[
        256,
        256,
        3]):
    if model_name == "auto_encoder":
        model = DeepConvAutoEncoder(
            epoch=epochs,
            learning_rate=training['lr'],
            white_bk=training['bk'],
            batch_size=batch_size,
            loss=training['loss'])
        return model
    if model_name == "GAN":
        model = K_DCGAN(
            epoch=epochs,
            img_shape=img_dim,
        )
        return model


def train(*args):
    pass
