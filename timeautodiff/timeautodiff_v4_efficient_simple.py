import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import tqdm.notebook
import gc
import random
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam
import math
from rich.progress import Progress
import os 
import wandb
import timeautodiff.processing_simple as processing

device = 'cuda' if torch.cuda.is_available() else 'cpu'

'''
Improvements from v4 efficient/ Overall features: 
- simplified the processing.

Some notes:
1.  The pipeline implicitly assumes that the data is already normalized. 
    --> Sinusoidal embeddings explode or randomize:
    angles = 2**torch.arange(8) * π * x_nums.unsqueeze(-1)
    If the raw numeric values aren't in a small, consistent range (e.g. [0, 1]), those angles will be huge and the sines/cosines will look like random noise.
    --> Sigmoid outputs expect [0, 1] targets: 
    decoded_nums = torch.sigmoid(self.nums_linear(latent))
    the final numeric decoder assumes the ground-truth nums are also in [0, 1] so that the MSE loss makes sense
    --> MMD kernel bandwidth is sensitive to scale
    The Guassian kernel will collapse to 0 or 1 if the feature scales vary widly.
2. No data loader
3. The pipeline assumes that the time-seris is of fixed length.
4. the pipeline has been updated to return the generated data into the original order of the data

'''
################################################################################################################
# MMD Loss Function

def compute_mmd(x, y):
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of samples.
    """
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return x_kernel.mean() + y_kernel.mean() - 2 * xy_kernel.mean()

def compute_kernel(x, y):
    """
    Gaussian kernel between two sets of samples.
    """
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(-1)
    seq_len = x.size(1)
    feature_dim = x.size(2)
    tiled_x = x.unsqueeze(1).expand(x_size, y_size, seq_len, feature_dim)
    tiled_y = y.unsqueeze(0).expand(x_size, y_size, seq_len, feature_dim)
    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=(-1, -2)) / feature_dim)

################################################################################################################
# Consistency Regularization

# def consistency_regularization(x, perturbed_x, model):
#     """
#     Encourage model output consistency between original and perturbed inputs.
#     """
#     original_output, _, _, _ = model(x)
#     perturbed_output, _, _, _ = model(perturbed_x)
#     return F.mse_loss(original_output, perturbed_output)


def consistency_regularization(x, perturbed_x, model, is_autoencoder=True, t=None, cond=None, time_info=None):
    """
    Encourage model output consistency between original and perturbed inputs.
    """
    if is_autoencoder:
        # For autoencoder models (like DeapStack)
        original_output = model(x)[0]  # Extract the decoded output dictionary
        perturbed_output = model(perturbed_x)[0]  # Extract the decoded output dictionary
        consistency_loss = 0

        # Calculate consistency loss for each type of output separately
        for key in original_output.keys():
            if key in perturbed_output:
                if key == 'bins':
                    # Use Binary Cross-Entropy for binary outputs
                    consistency_loss += F.binary_cross_entropy_with_logits(
                        original_output[key], perturbed_output[key]
                    )
                elif key == 'cats':
                    # Use Cross-Entropy for categorical outputs
                    for orig_cat, pert_cat in zip(original_output[key], perturbed_output[key]):
                        consistency_loss += F.cross_entropy(orig_cat, pert_cat.argmax(dim=-1))
                elif key == 'nums':
                    # Use Mean Squared Error for numerical outputs
                    consistency_loss += F.mse_loss(original_output[key], perturbed_output[key])
    else:
        # For BiRNN_score model (diffusion model)
        if t is None or cond is None or time_info is None:
            raise ValueError("For BiRNN_score model, t, cond, and time_info must be provided for consistency regularization.")

        # Use the same time, condition, and time information for both original and perturbed inputs
        original_output = model(x, t, t, cond, time_info)
        perturbed_output = model(perturbed_x, t, t, cond, time_info)
        consistency_loss = F.mse_loss(original_output, perturbed_output)

    return consistency_loss

################################################################################################################
# Optimize compute_sine_cosine function to use less memory
def compute_sine_cosine(v, num_terms):
    num_terms = torch.tensor(num_terms).to(device)
    v = v.to(device)
    
    # Process in chunks to reduce memory usage
    chunk_size = 1000
    result_chunks = []
    
    for i in range(0, v.shape[0], chunk_size):
        chunk = v[i:i+chunk_size]
        angles = 2**torch.arange(num_terms).float().to(device) * torch.tensor(math.pi).to(device) * chunk.unsqueeze(-1)
        sine_values = torch.sin(angles)
        cosine_values = torch.cos(angles)
        
        sine_values = sine_values.view(*sine_values.shape[:-2], -1)
        cosine_values = cosine_values.view(*cosine_values.shape[:-2], -1)
        
        result = torch.cat((sine_values, cosine_values), dim=-1)
        result_chunks.append(result)
        
        # Clear unnecessary tensors
        del angles, sine_values, cosine_values
        torch.cuda.empty_cache()
    
    return torch.cat(result_chunks, dim=0)

################################################################################################################
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.RNN = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, d_last_states = self.RNN(x)
        y_hat_logit = self.fc(d_last_states[-1])
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat

################################################################################################################
class Embedding_data_auto(nn.Module):
    def __init__(self, input_size, emb_dim, n_bins, n_cats, n_nums, cards):
        super().__init__()
        self.input_size = input_size
        self.emb_dim = emb_dim
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards
        self.n_disc = self.n_bins + self.n_cats
        self.num_categorical_list = [2]*self.n_bins + self.cards
        
        if self.n_disc != 0:
            self.embeddings_list = nn.ModuleList([
                nn.Embedding(num_categories, emb_dim)
                for num_categories in self.num_categorical_list
            ])
        
        if self.n_nums != 0:
            self.mlp_nums = nn.Sequential(
                nn.Linear(16 * n_nums, 16 * n_nums),
                nn.SiLU(),
                nn.Linear(16 * n_nums, 16 * n_nums)
            )
            
        self.mlp_output = nn.Sequential(
            nn.Linear(emb_dim * self.n_disc + 16 * n_nums, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, input_size)
        )
        
    def process_chunk(self, x, chunk_size=32):
        B, L, _ = x.shape
        device = x.device
        x_emb_chunks = []
        
        for i in range(0, B, chunk_size):
            chunk = x[i:i+chunk_size].to(device)
            x_disc = chunk[:,:,0:self.n_disc].long()
            x_nums = chunk[:,:,self.n_disc:self.n_disc+self.n_nums]
            
            if self.n_disc != 0:
                emb_list = []
                for j, embedding in enumerate(self.embeddings_list):
                    emb = embedding(x_disc[:,:,j])
                    emb_list.append(emb)
                x_disc_emb = torch.cat(emb_list, dim=2)
                del emb_list
            else:
                x_disc_emb = torch.tensor([], device=device)
            
            if self.n_nums != 0:
                angles = 2**torch.arange(8, device=device).float() * math.pi * x_nums.unsqueeze(-1)
                sines = torch.sin(angles)
                cosines = torch.cos(angles)
                
                trig_values = torch.cat([
                    sines.reshape(*sines.shape[:-2], -1),
                    cosines.reshape(*cosines.shape[:-2], -1)
                ], dim=-1)
                del sines, cosines, angles
                
                x_nums_emb = self.mlp_nums(trig_values)
                x_emb = torch.cat([x_disc_emb, x_nums_emb], dim=2)
                del trig_values, x_nums_emb
            else:
                x_emb = x_disc_emb
            
            x_emb = self.mlp_output(x_emb)
            x_emb_chunks.append(x_emb)
            
            del x_disc, x_nums, x_disc_emb
            torch.cuda.empty_cache()
            
        return torch.cat(x_emb_chunks, dim=0)

    def forward(self, x):
        self.to(x.device)  # Ensure model is on same device as input
        return self.process_chunk(x)
    
class Embedding_data_diff(nn.Module):
    #   processes categorical, binary, and numerical inputs by embedding them
    #   into dense vectors and passing numerical data through a Multi-Layer Perceptron (MLP).
    
    def __init__(self, input_size, emb_dim, n_bins, n_cats, n_nums, cards):
        super().__init__()
        
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.cards = cards
        
        self.n_disc = self.n_bins + self.n_cats
        self.num_categorical_list = [2]*self.n_bins + self.cards
        
        if self.n_disc != 0:
            # Create a list to store individual embeddings
            self.embeddings_list = nn.ModuleList()
            
            # Create individual embeddings for each variable
            for num_categories in self.num_categorical_list:
                embedding = nn.Embedding(num_categories, emb_dim)
                self.embeddings_list.append(embedding)
        
        if self.n_nums != 0:
            self.mlp_nums = nn.Sequential(nn.Linear(n_nums, n_nums),  # this should be 16 * n_nums, 16 * n_nums
                                          nn.SiLU(),
                                          nn.Linear(n_nums, n_nums))
            
        self.mlp_output = nn.Sequential(nn.Linear(emb_dim * self.n_disc + n_nums, emb_dim), # this should be 16 * n_nums, 16 * n_nums
                                       nn.ReLU(),
                                       nn.Linear(emb_dim, emb_dim))
        
    def forward(self, x):
        #TODO 
        x_disc = x[:,:,0:self.n_disc].long().to(device)
        x_nums = x[:,:,self.n_disc:self.n_disc+self.n_nums].to(device)
        
        x_emb = torch.Tensor().to(device)
        
        # Binary + Discrete Variables
        if self.n_disc != 0:
            variable_embeddings = [embedding(x_disc[:,:,i]) for i, embedding in enumerate(self.embeddings_list)]
            x_disc_emb = torch.cat(variable_embeddings, dim=2)
            x_emb = x_disc_emb

        # Numerical Variables
        if self.n_nums != 0:
            #x_nums = compute_sine_cosine(x_nums, num_terms=8)
            x_nums_emb = self.mlp_nums(x_nums)
            x_emb = torch.cat([x_emb, x_nums_emb], dim=2)
        
        final_emb = self.mlp_output(x_emb)
        
        return final_emb
################################################################################################################
#def get_torch_trans(heads = 8, layers = 1, channels = 64):
#    encoder_layer = nn.TransformerEncoderLayer(d_model = channels, nhead = heads, dim_feedforward=64, activation = "gelu")
#    return nn.TransformerEncoder(encoder_layer, num_layers = layers)

#class Transformer_Block(nn.Module):
#    def __init__(self, channels):
#        super().__init__()
#        self.channels = channels
        
#        self.conv_layer1 = nn.Conv1d(1, self.channels, 1)
#        self.feature_layer = get_torch_trans(heads = 8, layers = 1, channels = self.channels)
#        self.conv_layer2 = nn.Conv1d(self.channels, 1, 1)
    
#    def forward_feature(self, y, base_shape):
#        B, channels, L, K = base_shape
#        if K == 1:
#            return y.squeeze(1)
#        y = y.reshape(B, channels, L, K).permute(0, 2, 1, 3).reshape(B*L, channels, K)
#        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
#        y = y.reshape(B, L, channels, K).permute(0, 2, 1, 3)
#        return y
    
#    def forward(self, x):
#        x = x.unsqueeze(1)
#        B, input_channel, K, L = x.shape
#        base_shape = x.shape

#        x = x.reshape(B, input_channel, K*L)       
        
#        conv_x = self.conv_layer1(x).reshape(B, self.channels, K, L)
#        x = self.forward_feature(conv_x, conv_x.shape)
#        x = self.conv_layer2(x.reshape(B, self.channels, K*L)).squeeze(1).reshape(B, K, L)
        
#        return x

################################################################################################################
class DeapStack(nn.Module):
    def __init__(self, channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, input_size,
                 hidden_size, num_layers, cat_emb_dim, time_dim, lat_dim, column_order):
        super().__init__()
        # TODO remove time_dim and time_encode from AE
        self.Emb = Embedding_data_auto(input_size, cat_emb_dim, n_bins, n_cats, n_nums, cards)
        self.time_encode = nn.Sequential(nn.Linear(time_dim, input_size),
                                         nn.ReLU(),
                                         nn.Linear(input_size, input_size))
        
        self.encoder_mu = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.encoder_logvar = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        self.fc_mu = nn.Linear(hidden_size, lat_dim)
        self.fc_logvar = nn.Linear(hidden_size, lat_dim)

        self.decoder_mlp = nn.Sequential(nn.Linear(lat_dim, hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(hidden_size, hidden_size))
        # Save configuration parameters
        self.config = {
            'channels': channels,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'n_bins': n_bins,
            'n_cats': n_cats,
            'n_nums': n_nums,
            'cards': cards,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'cat_emb_dim': cat_emb_dim,
            'time_dim': time_dim,
            'lat_dim': lat_dim,
            'column_order': column_order
        }
        
        self.lat_dim = lat_dim
        self.channels = channels
        self.n_bins = n_bins
        self.n_cats = n_cats
        self.n_nums = n_nums
        self.column_order = column_order    
        self.cards = cards
        self.disc = self.n_bins + self.n_cats
        self.sigmoid = torch.nn.Sigmoid()
        
        self.bins_linear = nn.Linear(hidden_size, n_bins) if n_bins else None
        self.cats_linears = nn.ModuleList([nn.Linear(hidden_size, card) for card in cards]) if n_cats else None 
        self.nums_linear = nn.Linear(hidden_size, n_nums) if n_nums else None

    def save_model(self, save_path):
        """
        Save both model state and configuration parameters
        
        Args:
            save_path (str): Path to save the model (without extension)
        """
        # Save model state
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, f"{save_path}.pt")
    

    
    
    @classmethod
    def load_model(cls, load_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load model from saved state and configuration
        
        Args:
            load_path (str): Path to the saved model (without extension)
            device (str): Device to load the model to
            
        Returns:
            DeapStack: Loaded model instance
        """
        # Load saved state
        checkpoint = torch.load(load_path, map_location=device)
        
        # Create new model instance with saved configuration
        model = cls(**checkpoint['config'])
        
        # Load state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        model = model.to(device)
        return model

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encoder(self, x):
        x = self.Emb(x)
        
        mu_z, _ = self.encoder_mu(x)
        logvar_z, _ = self.encoder_logvar(x)
        
        mu_z = self.fc_mu(mu_z); logvar_z = self.fc_logvar(logvar_z)
        emb = self.reparametrize(mu_z, logvar_z)
        
        return emb, mu_z, logvar_z

    def decoder(self, latent_feature):
        decoded_outputs = dict()
        latent_feature = self.decoder_mlp(latent_feature)
        
        B, L, K = latent_feature.shape
        
        if self.bins_linear:
            decoded_outputs['bins'] = self.bins_linear(latent_feature)

        if self.cats_linears:
            decoded_outputs['cats'] = [linear(latent_feature) for linear in self.cats_linears]

        if self.nums_linear:
            decoded_outputs['nums'] = self.sigmoid(self.nums_linear(latent_feature))

        return decoded_outputs

    def forward(self, x):
        emb, mu_z, logvar_z = self.encoder(x)
        outputs = self.decoder(emb)
        return outputs, emb, mu_z, logvar_z
    
def auto_loss(inputs, reconstruction, n_bins, n_nums, n_cats, beta, cards):
    """ Calculating the loss for DAE network.
        BCE for masks and reconstruction of binary inputs.
        CE for categoricals.
        MSE for numericals.
        reconstruction loss is weighted average of mean reduction of loss per datatype.
        mask loss is mean reduced.
        final loss is weighted sum of reconstruction loss and mask loss.
    """
    B, L, K = inputs.shape
    # assumes the data is in the order of bins, cats, nums
    bins = inputs[:,:,0:n_bins]
    cats = inputs[:,:,n_bins:n_bins+n_cats].long()
    nums = inputs[:,:,n_bins+n_cats:n_bins+n_cats+n_nums]

    #reconstruction_losses = dict()
    disc_loss = 0; num_loss = 0;
    
    if 'bins' in reconstruction:
        disc_loss += F.binary_cross_entropy_with_logits(reconstruction['bins'], bins)

    if 'cats' in reconstruction:
        cats_losses = []
        for i in range(len(reconstruction['cats'])):
            cats_losses.append(F.cross_entropy(reconstruction['cats'][i].reshape(B*L, cards[i]), \
                                               cats[:,:,i].unsqueeze(2).reshape(B*L, 1).squeeze(1)))
        disc_loss += torch.stack(cats_losses).mean()

    if 'nums' in reconstruction:
        num_loss = F.mse_loss(reconstruction['nums'], nums)

    #reconstruction_loss = torch.stack(list(reconstruction_losses.values())).mean()

    return disc_loss, num_loss
################################################################################################################
def evaluate_model(ae, processed_data, device, max_gpu_chunk=100):
    outputs = []
    latents = []
    mus = []
    logvars = []
    
    # Process data in small batches
    with torch.no_grad():
        for i in range(0, processed_data.shape[0], max_gpu_chunk):
            chunk = processed_data[i:i+max_gpu_chunk].to(device)
            out, lat, mu, logvar = ae(chunk)
            
            # Move results to CPU and append
            outputs.append({k: v.cpu() for k, v in out.items()})
            latents.append(lat.cpu())
            mus.append(mu.cpu())
            logvars.append(logvar.cpu())
            
            # Clear memory
            del chunk, out, lat, mu, logvar
            torch.cuda.empty_cache()
    
    # Combine results
    final_latents = torch.cat(latents, dim=0)
    final_mus = torch.cat(mus, dim=0)
    final_logvars = torch.cat(logvars, dim=0)
    
    return outputs, final_latents, final_mus, final_logvars

# Train Autoencoder with MMD and Consistency Regularization
# def train_autoencoder(real_df, processed_data, channels, hidden_size, num_layers, lr, weight_decay, 
#                      n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, 
#                      device, dir, checkpoints=False, mmd_weight = 0.1, consistency_weight = 0.1):
#     parser = pce.DataFrameParser().fit(real_df, threshold)
#     data = parser.transform()
#     data = torch.tensor(data.astype('float32')).unsqueeze(0)
    
#     datatype_info = parser.datatype_info()
#     n_bins = datatype_info['n_bins']
#     n_cats = datatype_info['n_cats']
#     n_nums = datatype_info['n_nums']
#     cards = datatype_info['cards']

def train_autoencoder(processed_data, channels, hidden_size, num_layers, lr, weight_decay, 
                     n_epochs, batch_size, min_beta, max_beta, emb_dim, time_dim, lat_dim, 
                     device, dir, checkpoints=False, mmd_weight = 0.1, consistency_weight = 0.1, use_wandb=False):
    N, seq_len, input_size = processed_data.shape
    
        
    parser = processing.DataFrameParser().fit(pd.DataFrame(processed_data.reshape(-1,input_size)), 1, fit_encoders=False)
    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']
    column_order = parser._column_order
    ae = DeapStack(channels, batch_size, seq_len, n_bins, n_cats, n_nums, cards, 
                  input_size, hidden_size, num_layers, emb_dim, time_dim, lat_dim, column_order).to(device)
    
    optimizer_ae = Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    processed_data = processed_data[:,:,column_order].to(device)

    losses = []
    recons_loss = []
    KL_loss = []
    beta = max_beta
    
    lambd = 0.7
    best_train_loss = float('inf')
    all_indices = list(range(N))

    def process_batch(batch_data):
        outputs, emb, mu_z, logvar_z = ae(batch_data)
        disc_loss, num_loss = auto_loss(batch_data, outputs, n_bins, n_nums, n_cats, beta, cards)
        temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()
        loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
        
        # MMD Loss
        if mmd_weight > 0:
            mmd_loss = compute_mmd(emb, torch.randn_like(emb))
        else: 
            mmd_loss = 0
        # Consistency Regularization
        if consistency_weight > 0:
            perturbed_inputs = batch_data + 0.1 * torch.randn_like(batch_data).to(device)
            consistency_loss = consistency_regularization(batch_data, perturbed_inputs, ae)
        else:
            consistency_loss = 0 
        loss_Auto = num_loss + disc_loss + beta * loss_kld + mmd_weight * mmd_loss + consistency_weight * consistency_loss
        return loss_Auto, disc_loss, num_loss, loss_kld,mmd_loss, consistency_loss
    
    def save_checkpoint(epoch):
        checkpoint_dir = os.path.join(dir, "autoencoder_checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        ae.save_model(os.path.join(checkpoint_dir, f"model_epoch_{epoch}"))
        
        chunk_size = 32
        latent_features_list = []
        
        for i in range(0, processed_data.shape[0], chunk_size):
            chunk = processed_data[i:i+chunk_size]
            with torch.no_grad():
                _, latent, _, _ = ae(chunk)
                latent_features_list.append(latent.cpu())
            torch.cuda.empty_cache()
        
        latent_features = torch.cat(latent_features_list, dim=0)
        torch.save(latent_features, os.path.join(checkpoint_dir, f"latent_features_epoch_{epoch}.pt"))
        del latent_features, latent_features_list
        torch.cuda.empty_cache()
    '''
    Total examples seen = n_epochs * batch_size = 50 000 * 100 = 5 000 000 samples.
    If the dataset has, say, 100 000 examples, that’s the equivalent of 50 full
    passes (i.e. 50 “effective epochs”)—just scattered randomly across those 50 000 updates.
    '''
    with Progress() as progress:
        training_task = progress.add_task("[red]Training...", total=n_epochs)

        for epoch in range(n_epochs):
            ae.train()  
            batch_indices = random.sample(all_indices, batch_size)
            batch_data = processed_data[batch_indices]

            optimizer_ae.zero_grad()
            loss_Auto,disc_loss, num_loss, loss_kld, mmd_loss, consistency_loss = process_batch(batch_data)

            loss_Auto.backward()
            optimizer_ae.step()

            progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {loss_Auto.item():.4f}")
            if checkpoints and epoch % 10000 == 0 and epoch != 0 and epoch != n_epochs - 1:
                save_checkpoint(epoch)
            if use_wandb:
                wandb.log({
                    "auto_epoch": epoch, 
                    "auto_loss": loss_Auto.item() if hasattr(loss_Auto, 'item') else loss_Auto, 
                    "disc_loss": disc_loss.item() if hasattr(disc_loss, 'item') else disc_loss, 
                    "num_loss": num_loss.item() if hasattr(num_loss, 'item') else num_loss, 
                    "KL_loss": loss_kld.item() if hasattr(loss_kld, 'item') else loss_kld,
                    "mmd_loss": mmd_loss.item() if hasattr(mmd_loss, 'item') else mmd_loss,
                    "consistency_loss": consistency_loss.item() if hasattr(consistency_loss, 'item') else consistency_loss
                })
            
            if loss_Auto < best_train_loss:
                best_train_loss = loss_Auto
                patience = 0
            else:
                patience += 1
                if patience == 10 and beta > min_beta:
                    beta = beta * lambd
            # progress.update(training_task, advance=1)


    # Final evaluation
    ae.eval()
    outputs, final_latents, final_mus, final_logvars = evaluate_model(ae, processed_data, device)
    
    return (ae, final_latents, outputs, losses, recons_loss, final_mus, final_logvars)

    
################################################################################################################

################################################################################################################

# def load_autoencoder(filepath, real_df, processed_data, channels, hidden_size, num_layers,  batch_size, threshold, emb_dim, time_dim, lat_dim):
#     parser = pce.DataFrameParser().fit(real_df, threshold)
#     data = parser.transform()
#     data = torch.tensor(data.astype('float32')).unsqueeze(0)
        
#     datatype_info = parser.datatype_info()
#     n_bins = datatype_info['n_bins']; n_cats = datatype_info['n_cats']
#     n_nums = datatype_info['n_nums']; cards = datatype_info['cards']
#     # Create a dictionary with datatype information


################################################################################################################
def get_betas(steps):
    """	
    linear schedule of beta values that control the noise level across diffusion steps.
    
    """	
    
    beta_start, beta_end = 1e-4, 0.2
    diffusion_ind = torch.linspace(0, 1, steps).to(device)
    return beta_start * (1 - diffusion_ind) + beta_end * diffusion_ind

diffusion_steps = 100
betas = get_betas(diffusion_steps)
alphas = torch.cumprod(1 - betas, dim=0) #cumulative product of (1 - betas),representing the proportion of the original data remaining after noise is added.

def get_gp_covariance(t):
    s = t - t.transpose(-1, -2)
    diag = torch.eye(t.shape[-2]).to(t) 
    return diag 

def add_noise(x, t, i):
    """
    adds Gaussian noise to the input data,modulated by the beta and alpha parameters,
    to create noisy versions of the data over time.This function is critical for the
    forward diffusionprocess, gradually corrupting the data until it becomes pure noise.
    
    
    x: Clean data sample, shape [B, S, D]
    t: Times of observations, shape [B, S, 1]
    i: Diffusion step, shape [B, S, 1]
    """
    noise_gaussian = torch.randn_like(x)
    
    cov = get_gp_covariance(t)
    L = torch.linalg.cholesky(cov)
    
    noise = L @ noise_gaussian
    
    alpha = alphas[i.long()].to(x)
    x_noisy = torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise
    
    return x_noisy, noise

#####################################################################################################################

from typing import List, Callable
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PositionalEncoding(nn.Module):
    """
    encode time steps using a combination of linear and periodic transformations.
    This encoding helps the model understand and utilize the temporal dynamics in the data.
    Used for encoding the timesteps t and the diffusion steps n.
    """
    def __init__(self, dim: int, max_value: float):
        super().__init__()
        self.max_value = max_value

        linear_dim = dim // 2
        periodic_dim = dim - linear_dim

        self.scale = torch.exp(-2 * torch.arange(0, periodic_dim).float() * math.log(self.max_value) / periodic_dim)
        self.shift = torch.zeros(periodic_dim)
        self.shift[::2] = 0.5 * math.pi

        self.linear_proj = nn.Linear(1, linear_dim)

    def forward(self, t):
        periodic = torch.sin(t * self.scale.to(t) + self.shift.to(t))
        linear = self.linear_proj(t / self.max_value)
        return torch.cat([linear, periodic], -1)

class FeedForward(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, activation: Callable=nn.ReLU(), final_activation: Callable=None):
        super().__init__()

        hidden_dims = hidden_dims[:]
        hidden_dims.append(out_dim)

        layers = [nn.Linear(in_dim, hidden_dims[0])]

        for i in range(len(hidden_dims) - 1):
            layers.append(activation)
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

        if final_activation is not None:
            layers.append(final_activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
# https://datascience.stackexchange.com/questions/121548/how-to-make-an-rnn-model-in-pytorch-that-has-a-custom-hidden-layers-and-that-i
class BiRNN_score(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, diffusion_steps,
                 cond_dim, time_dim, emb_dim, n_bins, n_cats, n_nums, cards,column_order):
        super(BiRNN_score, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Save configuration parameters
        self.config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'diffusion_steps': diffusion_steps,
            'cond_dim': cond_dim,
            'time_dim': time_dim,
            'emb_dim': emb_dim,
            'n_bins': n_bins,
            'n_cats': n_cats,
            'n_nums': n_nums,
            'cards': cards,
            'column_order': column_order
        }
        
        self.input_proj = FeedForward(input_size, [], hidden_size)
        self.t_enc = PositionalEncoding(hidden_size, max_value=1)
        self.i_enc = PositionalEncoding(hidden_size, max_value=diffusion_steps) 
        self.proj = FeedForward(4 * hidden_size, [], hidden_size, final_activation=nn.ReLU())
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.fc = nn.Linear(2 * hidden_size, input_size)
        
        self.Emb = Embedding_data_diff(input_size, emb_dim, n_bins, n_cats, n_nums, cards)
        self.cond_lstm = nn.LSTM(emb_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.cond_output = nn.Linear(2*hidden_size, hidden_size)
        
        self.time_encode = nn.Sequential(nn.Linear(time_dim, hidden_size),
                                 nn.ReLU(),
                                 nn.Linear(hidden_size, hidden_size))

    def save_model(self, save_path):
        """
        Save both model state and configuration parameters
        
        Args:
            save_path (str): Path to save the model (without extension)
        """
        # Save model state and config
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, f"{save_path}.pt")
    
    @classmethod
    def load_model(cls, load_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Load model from saved state and configuration
        
        Args:
            load_path (str): Path to the saved model (without extension)
            device (str): Device to load the model to
            
        Returns:
            BiRNN_score: Loaded model instance
        """
        # Load saved state
        checkpoint = torch.load(load_path, map_location=device)
        
        # Create new model instance with saved configuration
        model = cls(**checkpoint['config'])
        
        # Load state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        model = model.to(device)
        return model

    def forward(self, x, t, i, cond = None, time_info = None):
        shape = x.shape

        x = x.view(-1, *shape[-2:])
        t = t.view(-1, shape[-2], 1)
        i = i.view(-1, shape[-2], 1)
        
        x = self.input_proj(x)
        t = self.t_enc(t)
        i = self.i_enc(i)
        time_info = self.time_encode(time_info)

        if cond is not None:            
            cond_out, _ = self.cond_lstm(self.Emb(cond))
            x = self.proj(torch.cat([x + self.cond_output(cond_out), t, i, time_info], -1))    
        else:
            x = self.proj(torch.cat([x, t, i, time_info], -1))
            
        out, _ = self.lstm(x)
        output = self.layer_norm(out)
        final_out = self.fc(output)
        
        return final_out

#####################################################################################################################
class EMA:
    """
    An EMA model to help stabilize the training updates.
    by maintaining a smoothed version of the model parameters.
    """
    def __init__(self, beta):
        self.beta = beta
        self.step = 0
    
    def update_average(self, old, new):
        return self.beta * old + (1-self.beta) * new
    
    def update_model_average(self, ema_model, model):
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weight, new_weight)
    
    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())
    
    def step_ema(self, ema_model, model, step_start_ema = 2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

#####################################################################################################################
def get_loss(model, x, t, cond = None, time_info = None):
    i = torch.randint(0, diffusion_steps, size=(x.shape[0],))
    i = i.view(-1, 1, 1).expand_as(x[...,:1]).to(x)
    #Adding noise to clean data based on the current diffusion step.
    x_noisy, noise = add_noise(x, t, i) 
    #Having the model predict this noise based on the noised data.
    pred_noise = model(x_noisy, t, i, cond, time_info)
    #Calculating the mean squared error between the predicted and actual noise.
    loss = (pred_noise - noise)**2
    return torch.mean(loss)

#####################################################################################################################
import copy
import tqdm.notebook
import random
import wandb

# Train Diffusion Model with MMD and Consistency Regularization
def train_diffusion(latent_features, cond_tensor, time_info, hidden_dim, num_layers, 
                    diffusion_steps, n_epochs, out_dir, checkpoints=False, num_classes=None,
                    mmd_weight = 0.1, consistency_weight = 0.1, device = 'cuda', use_wandb=False):
    emb_dim = 128
    input_size = latent_features.shape[2]
    time_dim = time_info.shape[2]
    cond_dim = cond_tensor.shape[2]
    
    parser = processing.DataFrameParser().fit(pd.DataFrame(cond_tensor.reshape(-1,cond_dim)), 1, fit_encoders=False)
    datatype_info = parser.datatype_info()
    n_bins = datatype_info['n_bins']
    n_cats = datatype_info['n_cats']
    n_nums = datatype_info['n_nums']
    cards = datatype_info['cards']   
    column_order = parser._column_order
    
    cond_tensor = cond_tensor[:,:,column_order].to(device)
    time_info = time_info.to(device)

    model = BiRNN_score(input_size, hidden_dim, num_layers, diffusion_steps,
                        cond_dim, time_dim, emb_dim, n_bins, n_cats, n_nums, cards, column_order).to(device)
    optim = torch.optim.Adam(model.parameters())
    ema = EMA(beta=0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    x = latent_features.detach().to(device)
    batch_size = diffusion_steps
    all_indices = list(range(len(latent_features)))
    
    # Add curriculum learning parameters
    warmup_steps = 1000
    curriculum_weight = lambda step: min(1.0, step / warmup_steps)
    if use_wandb:
        wandb.watch(model, log="all")

    with Progress() as progress:
        training_task = progress.add_task("[red]Training...", total=n_epochs)

        for epoch in range(n_epochs):
            batch_indices = random.sample(all_indices, batch_size)
            optim.zero_grad()
            
            # Apply curriculum weight
            curr_weight = curriculum_weight(epoch)
            t = torch.rand(diffusion_steps, latent_features.shape[1], 1).sort(1)[0].to(device) * curr_weight
            
            if cond_tensor is not None:
                loss = get_loss(model, x[batch_indices,:,:], t, cond_tensor[batch_indices,:,:], time_info[batch_indices,:,:])
            else:
                loss = get_loss(model, x[batch_indices,:,:], t, cond_tensor, time_info[batch_indices,:,:])
            
            # MMD Loss
            if mmd_weight > 0:
                mmd_loss = compute_mmd(x[batch_indices, :, :], torch.randn_like(x[batch_indices, :, :]))
            else:
                mmd_loss = 0
            # Consistency Regularization
            if consistency_weight > 0:
                perturbed_x = x[batch_indices, :, :] + 0.1 * torch.randn_like(x[batch_indices, :, :]).to(device)
                consistency_loss = consistency_regularization(x[batch_indices, :, :], perturbed_x, model,False, t, cond_tensor[batch_indices,:,:], time_info[batch_indices,:,:])
            else:
                consistency_loss = 0
                
            total_loss = loss + mmd_weight * mmd_loss + consistency_weight * consistency_loss
            total_loss.backward()
            optim.step()
            ema.step_ema(ema_model, model)
    
            progress.update(training_task, advance=1, description=f"Epoch {epoch}/{n_epochs} - Loss: {total_loss.item():.4f}")
            if use_wandb:
                wandb.log({"diff_epoch": epoch, "diff_loss": total_loss.item()})
            
            if checkpoints and epoch % 10000 == 0 and epoch != 0:
                checkpoint_dir = os.path.join(out_dir, "diff_checkpoints") 
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_model(os.path.join(checkpoint_dir, f"model_epoch_{epoch}"))
                # Also save EMA model
                ema_model.save_model(os.path.join(checkpoint_dir, f"ema_model_epoch_{epoch}"))
    
    # Save final models
    model.save_model(os.path.join(out_dir, "diffusion"))
    ema_model.save_model(os.path.join(out_dir, "diffusion_ema"))
    if use_wandb:
        wandb.finish()
    return model

#####################################################################################################################
@torch.no_grad()
def sample(t, B, T, F, model, cond, time_info):   
    x = torch.randn(B, T, F).to(device)
    
    time_info= time_info.to(device)
    column_order = model.config['column_order'] # static column order
    cond = cond[:,:,column_order].to(device) # make sure the cond is in the same order as the static column order

    for diff_step in reversed(range(0, diffusion_steps)):
        alpha = alphas[diff_step]
        beta = betas[diff_step]
        
        z = torch.randn(B, T, F).to(device)

        i = torch.Tensor([diff_step]).expand_as(x[...,:1]).to(device)
        
        cond_noise = model(x, t, i, cond, time_info)
        
        x = (1/(1 - beta).sqrt()) * (x - beta * cond_noise / (1 - alpha).sqrt()) + beta.sqrt() * z
    return x



