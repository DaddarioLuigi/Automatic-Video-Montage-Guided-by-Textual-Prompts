#!/usr/bin/env python3
"""
Example script demonstrating advanced NLP features:
- Advanced linguistic parsing with spaCy
- Semantic filtering with embeddings
- Text generation and summarization
"""

from src.prompt_processing import (
    AdvancedLinguisticParser,
    SemanticFilter,
    TextGenerator,
    PromptParser
)


def example_advanced_parsing():
    """Example of advanced linguistic parsing with spaCy."""
    print("=" * 60)
    print("EXAMPLE 1: Advanced Linguistic Parsing with spaCy")
    print("=" * 60)
    
    prompts = [
        "adding ingredients to the sandwich",
        "closing the box",
        "plating the dish"
    ]
    
    parser = AdvancedLinguisticParser(model_name="en_core_web_sm")
    parsed = parser.parse_prompts(prompts)
    
    print("\nParsed Information:")
    print(f"- Keywords: {parsed['aggregated']['keywords'][:10]}")
    print(f"- Verbs: {parsed['aggregated']['verbs']}")
    print(f"- Nouns: {parsed['aggregated']['nouns'][:10]}")
    
    if parsed['aggregated']['entities']:
        print(f"- Named Entities: {parsed['aggregated']['entities']}")
    
    if parsed['aggregated']['semantic_roles']:
        print("\nSemantic Roles (Subject-Verb-Object):")
        for role in parsed['aggregated']['semantic_roles'][:3]:
            print(f"  - {role['full_relation']}")
    
    # Example: Extract action phrases
    print("\nAction Phrases:")
    for prompt in prompts:
        phrases = parser.extract_action_phrases(prompt)
        print(f"  '{prompt}' -> {phrases}")


def example_semantic_filtering():
    """Example of semantic filtering with embeddings."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Semantic Filtering with Embeddings")
    print("=" * 60)
    
    prompts = ["adding ingredients", "closing box", "plating dish"]
    
    # Sample captions
    captions = [
        (0, "a person is putting ingredients into a sandwich"),
        (1, "someone is closing a container"),
        (2, "a chef is plating food on a dish"),
        (3, "a cat is sleeping on a couch"),
        (4, "a car is driving on the highway"),
    ]
    
    semantic_filter = SemanticFilter(model_name="all-MiniLM-L6-v2")
    semantic_filter.encode_prompts(prompts)
    
    # Filter captions using semantic similarity
    filtered = semantic_filter.filter_captions_semantic(
        captions,
        similarity_threshold=0.3
    )
    
    print(f"\nOriginal captions: {len(captions)}")
    print(f"Filtered captions: {len(filtered)}")
    print("\nFiltered Results (with similarity scores):")
    for idx, caption, score in filtered:
        print(f"  [{score:.3f}] Frame {idx}: {caption}")


def example_text_generation():
    """Example of text generation and summarization."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Text Generation and Summarization")
    print("=" * 60)
    
    text_generator = TextGenerator()
    
    # Example captions
    captions = [
        (0, "a person is adding various ingredients including lettuce, tomatoes, and cheese to a sandwich"),
        (1, "someone is carefully closing a cardboard box containing the prepared sandwich"),
        (2, "a chef is elegantly plating a beautifully arranged dish with garnishes and sauces")
    ]
    
    # Summarize individual captions
    print("\nSummarized Captions:")
    summarized = text_generator.summarize_captions(captions, max_length=20, min_length=5)
    for idx, (orig_idx, orig_cap) in enumerate(captions):
        print(f"  Original: {orig_cap}")
        print(f"  Summary:  {summarized[idx][1]}")
        print()
    
    # Create summary of all captions
    print("Combined Summary:")
    combined_summary = text_generator.summarize_caption_collection(captions)
    print(f"  {combined_summary}")
    
    # Generate alternative descriptions
    print("\nAlternative Descriptions:")
    original = "a person is adding ingredients to a sandwich"
    alternatives = text_generator.generate_alternative_descriptions(original, num_alternatives=3)
    print(f"  Original: {original}")
    for i, alt in enumerate(alternatives, 1):
        print(f"  Alt {i}:    {alt}")


def example_integrated_usage():
    """Example of integrated usage in the pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Integrated Usage with PromptParser")
    print("=" * 60)
    
    prompts = ["adding ingredients", "closing box"]
    captions = [
        (0, "a person is putting ingredients into a sandwich"),
        (1, "someone is closing a container"),
        (2, "a cat is sleeping"),
    ]
    
    # Create parser with advanced features enabled
    parser = PromptParser(
        use_advanced_parsing=True,
        use_semantic_filtering=True
    )
    
    # Parse prompts
    parsed = parser.parse(prompts)
    print(f"\nParsed prompts:")
    print(f"  Keywords: {list(parser.get_keywords())[:5]}")
    print(f"  Verbs: {list(parser.get_verbs())}")
    
    # Filter captions using semantic similarity
    filtered = parser.filter_captions(
        captions,
        semantic_threshold=0.3,
        use_semantic=True
    )
    
    print(f"\nFiltered {len(captions)} captions to {len(filtered)} using semantic similarity")


if __name__ == "__main__":
    try:
        example_advanced_parsing()
        example_semantic_filtering()
        example_text_generation()
        example_integrated_usage()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        print("\nNote: Make sure to install required models:")
        print("  python -m spacy download en_core_web_sm")
        print("\nThe sentence-transformers and transformers models")
        print("will be downloaded automatically on first use.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure you have installed the required dependencies:")
        print("  pip install -r requirements.txt")
        print("  python -m spacy download en_core_web_sm")
        import traceback
        traceback.print_exc()

