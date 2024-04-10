
import torch
from torch.autograd import Variable

#from models.vae import loss_function
from vae import loss_function

# For training VAE
def train(netEncoder, netDecoder, train_dataloader, optimizer1, optimizer2, num_train, beta, input_size):
    """
    For Training VAE
    Args:
        netEncoder : VAE Encoder
        netDecoder : VAE Decoder
        train_dataloader : Training dataloader
        optimizer1 : Training optimizer
        optimizer2 : Training optimizer
        num_train (int): The number of training samples 
        beta (float): The weight of KL divergence loss 
        input_size (int): the size of training data

    Returns:
        Executing VAE Training loss
    """
    netEncoder.train()
    netDecoder.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_dataloader):
        data = Variable(data)
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # recon_batch, mu, log_var = model(data)

        z, mu, log_var = netEncoder(data, input_size)
        recon_batch = netDecoder(z)
        loss = loss_function(recon_batch, data, mu, log_var, beta, input_size)

        loss.backward()

        train_loss += loss.data
        optimizer1.step()
        optimizer2.step()

    # train_loss /= len(df_train)
    train_loss /= num_train

    return train_loss


# For validation
def test(netEncoder, netDecoder, val_dataloader, num_val, beta, input_size):
    """
    For Training VAE
    Args:
        netEncoder : VAE Encoder
        netDecoder : VAE Decoder
        val_dataloader : Validation dataloader
        num_val (int): The number of validation samples 
        beta (float): The weight of KL divergence loss 
        input_size (int): the size of validation data

    Returns:
        Executing VAE validation loss
    """
    netEncoder.eval()
    netDecoder.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, data in enumerate(val_dataloader):
            data = Variable(data)
            z, mu, log_var = netEncoder(data, input_size)
            recon_batch = netDecoder(z)
            loss = loss_function(recon_batch, data, mu, log_var, beta, input_size)

            test_loss += loss.data

        # test_loss /= len(df_test)
        test_loss /= num_val

    return test_loss
