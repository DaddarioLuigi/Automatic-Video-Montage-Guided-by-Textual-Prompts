"""
Advanced Linguistic Parsing with spaCy

This module provides advanced NLP parsing capabilities including:
- POS tagging
- Named Entity Recognition (NER)
- Dependency parsing
- Semantic role extraction
"""

import spacy
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import warnings

# Suppress spaCy warnings
warnings.filterwarnings('ignore', category=UserWarning)


class AdvancedLinguisticParser:
    """Advanced linguistic parser using spaCy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the advanced parser.
        
        Args:
            model_name: spaCy model name (default: "en_core_web_sm")
                       Use "en_core_web_md" or "en_core_web_lg" for better accuracy
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Warning: {model_name} model not found. Installing...")
            print(f"Please run: python -m spacy download {model_name}")
            # Fallback to basic English model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    f"spaCy model not found. Please install with: "
                    f"python -m spacy download {model_name}"
                )
    
    def parse_prompt(self, prompt: str) -> Dict:
        """
        Parse a single prompt with advanced NLP techniques.
        
        Args:
            prompt: Textual prompt to parse
            
        Returns:
            Dictionary with parsed information:
            - tokens: List of tokens with POS tags
            - entities: Named entities found
            - verbs: Action verbs with their dependencies
            - nouns: Nouns with their modifiers
            - dependencies: Dependency parse tree
            - semantic_roles: Subject-verb-object relations
        """
        doc = self.nlp(prompt)
        
        # Extract tokens with POS tags
        tokens = [
            {
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'tag': token.tag_,
                'dep': token.dep_,
                'is_stop': token.is_stop
            }
            for token in doc
        ]
        
        # Extract named entities
        entities = [
            {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            }
            for ent in doc.ents
        ]
        
        # Extract verbs and their dependencies
        verbs = []
        for token in doc:
            if token.pos_ == "VERB":
                verb_info = {
                    'text': token.text,
                    'lemma': token.lemma_,
                    'tense': token.tag_,
                    'subjects': [],
                    'objects': [],
                    'modifiers': []
                }
                
                # Find subjects and objects
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
                        verb_info['subjects'].append({
                            'text': child.text,
                            'lemma': child.lemma_
                        })
                    elif child.dep_ in ["dobj", "pobj", "attr"]:
                        verb_info['objects'].append({
                            'text': child.text,
                            'lemma': child.lemma_
                        })
                    elif child.dep_ in ["advmod", "amod", "prep"]:
                        verb_info['modifiers'].append({
                            'text': child.text,
                            'lemma': child.lemma_,
                            'dep': child.dep_
                        })
                
                verbs.append(verb_info)
        
        # Extract nouns with their modifiers
        nouns = []
        for token in doc:
            if token.pos_ == "NOUN":
                noun_info = {
                    'text': token.text,
                    'lemma': token.lemma_,
                    'modifiers': [],
                    'determiners': []
                }
                
                for child in token.children:
                    if child.dep_ in ["amod", "compound", "nmod"]:
                        noun_info['modifiers'].append({
                            'text': child.text,
                            'lemma': child.lemma_,
                            'dep': child.dep_
                        })
                    elif child.dep_ == "det":
                        noun_info['determiners'].append(child.text)
                
                nouns.append(noun_info)
        
        # Extract semantic roles (subject-verb-object)
        semantic_roles = []
        for token in doc:
            if token.pos_ == "VERB":
                subject = None
                obj = None
                
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    elif child.dep_ in ["dobj", "pobj"]:
                        obj = child.text
                
                if subject or obj:
                    semantic_roles.append({
                        'subject': subject,
                        'verb': token.lemma_,
                        'object': obj,
                        'full_relation': f"{subject or '?'} {token.lemma_} {obj or '?'}"
                    })
        
        # Extract dependency tree
        dependencies = [
            {
                'head': token.head.text,
                'dep': token.dep_,
                'text': token.text
            }
            for token in doc
        ]
        
        return {
            'tokens': tokens,
            'entities': entities,
            'verbs': verbs,
            'nouns': nouns,
            'dependencies': dependencies,
            'semantic_roles': semantic_roles,
            'raw_text': prompt
        }
    
    def parse_prompts(self, prompts: List[str]) -> Dict:
        """
        Parse multiple prompts.
        
        Args:
            prompts: List of textual prompts
            
        Returns:
            Dictionary with aggregated parsing results
        """
        parsed_prompts = [self.parse_prompt(p) for p in prompts]
        
        # Aggregate information
        all_verbs = set()
        all_nouns = set()
        all_entities = set()
        all_keywords = set()
        
        for parsed in parsed_prompts:
            all_verbs.update(v['lemma'] for v in parsed['verbs'])
            all_nouns.update(n['lemma'] for n in parsed['nouns'])
            all_entities.update(e['text'] for e in parsed['entities'])
            # Extract important keywords (non-stop words, non-punctuation)
            all_keywords.update(
                t['lemma'] for t in parsed['tokens']
                if not t['is_stop'] and t['pos'] not in ['PUNCT', 'SPACE']
            )
        
        return {
            'parsed_prompts': parsed_prompts,
            'aggregated': {
                'verbs': list(all_verbs),
                'nouns': list(all_nouns),
                'entities': list(all_entities),
                'keywords': list(all_keywords),
                'semantic_roles': [sr for p in parsed_prompts for sr in p['semantic_roles']]
            },
            'original_prompts': prompts
        }
    
    def extract_action_phrases(self, prompt: str) -> List[str]:
        """
        Extract action phrases (verb + object) from prompt.
        
        Args:
            prompt: Textual prompt
            
        Returns:
            List of action phrases
        """
        doc = self.nlp(prompt)
        action_phrases = []
        
        for token in doc:
            if token.pos_ == "VERB":
                # Find direct object
                obj = None
                for child in token.children:
                    if child.dep_ == "dobj":
                        obj = child.text
                        break
                
                if obj:
                    action_phrases.append(f"{token.lemma_} {obj}")
                else:
                    action_phrases.append(token.lemma_)
        
        return action_phrases
    
    def get_synonyms_and_related(self, word: str, top_n: int = 5) -> List[str]:
        """
        Get synonyms and related words using word vectors.
        
        Args:
            word: Word to find synonyms for
            top_n: Number of similar words to return
            
        Returns:
            List of similar words
        """
        if word not in self.nlp.vocab:
            return []
        
        word_vector = self.nlp.vocab[word]
        if not word_vector.has_vector:
            return []
        
        # Find most similar words
        queries = [w for w in self.nlp.vocab if w.has_vector and w.is_lower == word.islower()]
        by_similarity = sorted(queries, key=lambda w: word_vector.similarity(w), reverse=True)
        
        similar_words = [
            w.text for w in by_similarity[:top_n + 1]
            if w.text != word and w.is_alpha
        ][:top_n]
        
        return similar_words

