"""
Semantic Disambiguation Module
Implements WSD algorithms from Chapter 10
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import re

from modules.bert_wsd import BERTSemanticDisambiguation

class LeskAlgorithm:
    """
    Lesk Algorithm for WSD (Chapter 10.1)
    s* = argmax Overlap(Gloss(s_i), C)
    """
    
    def __init__(self, sense_glosses: Dict[str, Dict[str, str]]):
        """
        Initialize Lesk algorithm
        
        Args:
            sense_glosses: {word: {sense: gloss_definition}}
        """
        self.glosses = sense_glosses
    
    def tokenize(self, text: str) -> set:
        """Simple tokenization"""
        return set(re.findall(r'\w+', text.lower()))
    
    def overlap(self, gloss: str, context: str) -> int:
        """Count overlapping words"""
        gloss_tokens = self.tokenize(gloss)
        context_tokens = self.tokenize(context)
        return len(gloss_tokens & context_tokens)
    
    def disambiguate(self, word: str, context: str) -> Tuple[str, float]:
        """
        Select best sense for word in context
        
        Args:
            word: Target word
            context: Surrounding text
            
        Returns:
            (best_sense, confidence_score)
        """
        if word not in self.glosses:
            return ("unknown", 0.0)
        
        senses = self.glosses[word]
        scores = {}
        
        for sense, gloss in senses.items():
            scores[sense] = self.overlap(gloss, context)
        
        if not scores or max(scores.values()) == 0:
            # Default to first sense
            return (list(senses.keys())[0], 0.5)
        
        best_sense = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = scores[best_sense] / total if total > 0 else 0.0
        
        return (best_sense, confidence)


class ContextEmbeddingWSD:
    """
    Embedding-based WSD using simple contextual similarity
    Simplified version of Transformer-based approach
    """
    
    def __init__(self, sense_definitions: Dict[str, Dict[str, str]]):
        """
        Args:
            sense_definitions: {word: {sense: definition}}
        """
        self.definitions = sense_definitions
    
    def simple_embedding(self, text: str) -> np.ndarray:
        """Create simple bag-of-words embedding"""
        words = re.findall(r'\w+', text.lower())
        vocab = set(words)
        # One-hot-like encoding (simplified)
        return np.array([1.0 if w in vocab else 0.0 for w in sorted(vocab)])
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Cosine similarity between vectors"""
        if len(v1) == 0 or len(v2) == 0:
            return 0.0
        # Pad to same length
        max_len = max(len(v1), len(v2))
        v1_padded = np.pad(v1, (0, max_len - len(v1)))
        v2_padded = np.pad(v2, (0, max_len - len(v2)))
        
        norm = np.linalg.norm(v1_padded) * np.linalg.norm(v2_padded)
        if norm == 0:
            return 0.0
        return np.dot(v1_padded, v2_padded) / norm
    
    def disambiguate(self, word: str, context: str) -> Tuple[str, float]:
        """
        Embedding-based sense selection
        
        Returns:
            (best_sense, similarity_score)
        """
        if word not in self.definitions:
            return ("unknown", 0.0)
        
        context_emb = self.simple_embedding(context)
        senses = self.definitions[word]
        
        similarities = {}
        for sense, definition in senses.items():
            sense_emb = self.simple_embedding(definition)
            similarities[sense] = self.cosine_similarity(context_emb, sense_emb)
        
        if not similarities:
            return (list(senses.keys())[0], 0.5)
        
        best_sense = max(similarities, key=similarities.get)
        return (best_sense, similarities[best_sense])


class SemanticAmbiguityCalculator:
    """
    Calculate semantic ambiguity score ψ(W, C)
    Based on Shannon Entropy (Chapter 4.2)
    
    ψ(W, C) = -Σ P(m_i|W,C) * log₂ P(m_i|W,C)
    """
    
    def __init__(self):
        """Initialize calculator"""
        pass
    
    def calculate_ambiguity(self, sense_probabilities: Dict[str, float]) -> float:
        """
        Calculate ambiguity from sense distribution
        
        Args:
            sense_probabilities: {sense: probability}
            
        Returns:
            Ambiguity score (0 = certain, higher = more ambiguous)
        """
        probs = np.array(list(sense_probabilities.values()))
        probs = probs / probs.sum()  # Normalize
        
        # Shannon entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        
        # Normalize by max entropy (uniform distribution)
        max_entropy = np.log2(len(probs)) if len(probs) > 1 else 0
        
        if max_entropy == 0:
            return 0.0
        
        return entropy / max_entropy
    
    def from_confidence(self, confidence: float) -> float:
        """
        Convert confidence score to ambiguity
        
        Args:
            confidence: Confidence in best sense (0-1)
            
        Returns:
            Ambiguity score (0-1)
        """
        return 1.0 - confidence


class IntegratedWSDSystem:
    """
    Complete WSD system for AIoT environments
    Combines multiple disambiguation methods
    """
    
    def __init__(self, sense_inventory: Dict[str, List[str]],
                 sense_glosses: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Args:
            sense_inventory: {word: [sense1, sense2, ...]}
            sense_glosses: Optional glossary
        """
        self.inventory = sense_inventory
        self.glosses = sense_glosses or self._create_default_glosses()
        
        self.lesk = LeskAlgorithm(self.glosses)
        self.embedding_wsd = ContextEmbeddingWSD(self.glosses)
        self.ambiguity_calc = SemanticAmbiguityCalculator()
        self.bert_wsd = BERTSemanticDisambiguation(sense_inventory)
        
    def _create_default_glosses(self) -> Dict[str, Dict[str, str]]:
        """Create default glossary from inventory"""
        glosses = {}
        for word, senses in self.inventory.items():
            glosses[word] = {sense: f"definition of {sense}" for sense in senses}
        return glosses
    
    def disambiguate(self, word, context, method = 'bert') -> Dict:
        """
        Disambiguate word in context
        
        Args:
            word: Target word
            context: Surrounding text
            method: 'lesk', 'embedding', or 'ensemble'
            
        Returns:
            {
                'sense': best_sense,
                'confidence': confidence_score,
                'ambiguity': ambiguity_score,
                'all_scores': {sense: score}
            }
        """
        if method == 'bert':
            return self.bert_wsd.disambiguate(word, context)
        
        elif method == 'lesk':
            sense, conf = self.lesk.disambiguate(word, context)
            all_scores = {sense: conf}
            
        elif method == 'embedding':
            sense, conf = self.embedding_wsd.disambiguate(word, context)
            all_scores = {sense: conf}
            
        else:  # ensemble
            sense1, conf1 = self.lesk.disambiguate(word, context)
            sense2, conf2 = self.embedding_wsd.disambiguate(word, context)
            
            # Vote or average
            if sense1 == sense2:
                sense = sense1
                conf = (conf1 + conf2) / 2
            else:
                sense = sense1 if conf1 > conf2 else sense2
                conf = max(conf1, conf2)
            
            all_scores = {sense1: conf1, sense2: conf2}
        
        # Calculate ambiguity
        if len(all_scores) > 1:
            ambiguity = self.ambiguity_calc.calculate_ambiguity(all_scores)
        else:
            ambiguity = self.ambiguity_calc.from_confidence(conf)
        
        return {
            'sense': sense,
            'confidence': conf,
            'ambiguity': ambiguity,
            'all_scores': all_scores
        }
    
    def process_command(self, command: str) -> Dict:
        """
        Process full operator command
        
        Returns:
            {
                'original': command,
                'disambiguated': [(word, sense), ...],
                'avg_confidence': float,
                'max_ambiguity': float
            }
        """
        words = re.findall(r'\w+', command.lower())
        disambiguated = []
        confidences = []
        ambiguities = []
        
        for word in words:
            if word in self.inventory:
                result = self.disambiguate(word, command)
                disambiguated.append((word, result['sense']))
                confidences.append(result['confidence'])
                ambiguities.append(result['ambiguity'])
        
        return {
            'original': command,
            'disambiguated': disambiguated,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'max_ambiguity': max(ambiguities) if ambiguities else 0.0
        }


# Industrial-specific sense inventories
INDUSTRIAL_SENSE_INVENTORY = {
    'line': ['assembly_line', 'electrical_line', 'hydraulic_line', 'queue_line'],
    'arm': ['robot_arm', 'mechanical_arm', 'human_arm'],
    'check': ['inspect_visually', 'test_functionally', 'query_status'],
    'stop': ['emergency_stop', 'normal_stop', 'pause'],
    'load': ['mechanical_load', 'cognitive_load', 'electrical_load'],
    'resistance': ['electrical_resistance', 'mechanical_resistance']
}

INDUSTRIAL_GLOSSES = {
    'line': {
        'assembly_line': 'production line conveyor manufacturing assembly',
        'electrical_line': 'power cable wire voltage current',
        'hydraulic_line': 'fluid pressure hose pipe hydraulic',
        'queue_line': 'waiting queue buffer tasks'
    },
    'arm': {
        'robot_arm': 'robotic manipulator actuator servo joint',
        'mechanical_arm': 'lever mechanism linkage',
        'human_arm': 'operator worker limb hand'
    },
    'check': {
        'inspect_visually': 'look examine visual inspection',
        'test_functionally': 'test verify operation function',
        'query_status': 'status report data system'
    },
    'stop': {
        'emergency_stop': 'emergency halt safety critical',
        'normal_stop': 'stop end complete finish',
        'pause': 'pause suspend temporary wait'
    },
    'load': {
        'mechanical_load': 'force weight mass pressure',
        'cognitive_load': 'mental workload thinking decision',
        'electrical_load': 'power current voltage consumption'
    }
}

if __name__ == "__main__":
    print("Semantic Disambiguation Test")
    print("=" * 50)
    
    # Create WSD system
    wsd = IntegratedWSDSystem(INDUSTRIAL_SENSE_INVENTORY, INDUSTRIAL_GLOSSES)
    
    # Test commands
    test_commands = [
        "check the assembly line",
        "stop the robot arm",
        "reduce the load on the system",
        "check electrical line voltage"
    ]
    
    for cmd in test_commands:
        result = wsd.process_command(cmd)
        print(f"\nCommand: '{result['original']}'")
        print(f"Disambiguated: {result['disambiguated']}")
        print(f"Confidence: {result['avg_confidence']:.2f}")
        print(f"Ambiguity: {result['max_ambiguity']:.2f}")
