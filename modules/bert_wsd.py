# modules/deep_wsd.py (NEW FILE)

import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

class BERTSemanticDisambiguation:
    """Real transformer-based WSD using BERT"""
    
    def __init__(self, sense_inventory):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased').to(self.device)
        
        # Classification head for sense selection
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, len(sense_inventory))
        ).to(self.device)
    
    def get_contextualized_embedding(self, word, context):
        """Get BERT embedding for word in context"""
        inputs = self.tokenizer(context, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.bert(**inputs)
            # Get embedding of target word
            word_tokens = self.tokenizer.encode(word, add_special_tokens=False)
            # Find word position in context
            token_embeddings = outputs.last_hidden_state[0]
            # Average if word is multiple tokens
            word_embedding = token_embeddings.mean(dim=0)
        
        return word_embedding
    
    def disambiguate(self, word, context):
        """Disambiguate word using BERT + classifier"""
        embedding = self.get_contextualized_embedding(word, context)
        
        # Classify sense
        logits = self.classifier(embedding.unsqueeze(0))
        probabilities = torch.softmax(logits, dim=1)
        
        predicted_sense = probabilities.argmax().item()
        confidence = probabilities.max().item()
        
        # Calculate ambiguity (Shannon entropy)
        probs = probabilities[0].cpu().numpy()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(probs))
        ambiguity = entropy / max_entropy if max_entropy > 0 else 0
        
        return {
            'sense_id': predicted_sense,
            'confidence': confidence,
            'ambiguity': ambiguity,
            'all_probabilities': probs
        }
    
    def fine_tune(self, training_data, epochs=10):
        """Fine-tune BERT for industrial WSD"""
        optimizer = torch.optim.AdamW(
            list(self.bert.parameters()) + list(self.classifier.parameters()),
            lr=2e-5
        )
        criterion = nn.CrossEntropyLoss()
        
        self.bert.train()
        self.classifier.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for word, context, true_sense in training_data:
                optimizer.zero_grad()
                
                embedding = self.get_contextualized_embedding(word, context)
                logits = self.classifier(embedding.unsqueeze(0))
                
                loss = criterion(logits, torch.tensor([true_sense]).to(self.device))
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(training_data):.4f}")