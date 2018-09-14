from src.auto_encoder import DeepConvAutoEncoder , l1_loss
from src.pix2pix_keras import K_DCGAN

def load_model(
    model_name,
    epochs,
    batch_size,
    learning_rate=1.e-6,
    white_bk=False,
    loss=l1_loss,
    img_dim=[256,256,3]):
    if model_name == "auto_encoder":
        model = DeepConvAutoEncoder(
            epoch=epochs,
            learning_rate=learning_rate,
            white_bk=white_bk,
            batch_size=batch_size,
            loss=loss)
        return model
    if model_name == "KDCGAN":
        assert isinstance(loss , list ) or isinstance(loss , tuple)
        model = K_DCGAN(
            epoch=epochs,
            img_shape=img_dim,
            
        )
        return model


def train(*args):
    pass


