import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VariationalAutoencoder(nn.Module):
    """VAE for unsupervised anomaly detection"""
    
    def __init__(self, input_dim=5, latent_dim=2):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 32)
        self.fc4 = nn.Linear(32, input_dim)
    
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = F.relu(self.fc3(z))
        return self.fc4(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class AnomalyDetector:
    """Complete anomaly detection system"""
    
    def __init__(self, input_dim=5, latent_dim=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae = VariationalAutoencoder(input_dim, latent_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=0.001)
        
        self.threshold = None
        self.scaler = None
    
    def vae_loss(self, recon_x, x, mu, logvar):
        """VAE loss = Reconstruction + KL divergence"""
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def train(self, normal_data, epochs=50):
        """Train on normal data only"""
        from sklearn.preprocessing import StandardScaler
        
        # Normalize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(normal_data)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.vae.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            
            recon_batch, mu, logvar = self.vae(X_tensor)
            loss = self.vae_loss(recon_batch, X_tensor, mu, logvar)
            
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()/len(X_tensor):.4f}")
        
        # Set threshold based on reconstruction error of normal data
        self.vae.eval()
        with torch.no_grad():
            recon, _, _ = self.vae(X_tensor)
            errors = torch.mean((X_tensor - recon)**2, dim=1)
            self.threshold = errors.mean() + 2 * errors.std()  # 2 sigma
    
    def detect_anomaly(self, data_point):
        """Detect if data point is anomalous"""
        self.vae.eval()
        
        # Normalize
        x_scaled = self.scaler.transform(data_point.reshape(1, -1))
        x_tensor = torch.FloatTensor(x_scaled).to(self.device)
        
        with torch.no_grad():
            recon, _, _ = self.vae(x_tensor)
            error = torch.mean((x_tensor - recon)**2)
        
        is_anomaly = error.item() > self.threshold.item()
        
        return {
            'is_anomaly': is_anomaly,
            'reconstruction_error': error.item(),
            'threshold': self.threshold.item(),
            'anomaly_score': error.item() / self.threshold.item()
        }
    
    def detect_batch(self, data):
        """Detect anomalies in batch"""
        self.vae.eval()
        
        X_scaled = self.scaler.transform(data)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            recon, _, _ = self.vae(X_tensor)
            errors = torch.mean((X_tensor - recon)**2, dim=1)
        
        anomalies = errors > self.threshold
        
        return anomalies.cpu().numpy(), errors.cpu().numpy()