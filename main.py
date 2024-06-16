import sys
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from bertopic.representation import KeyBERTInspired
from bertopic.representation import PartOfSpeech
from bertopic.representation import MaximalMarginalRelevance
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"File is empty: {csv_path}")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Error parsing the file: {csv_path}")
        sys.exit(1)

    if 'Summary' not in df.columns:
        print(f"Column 'Summary' not found in the file: {csv_path}")
        sys.exit(1)

    phrases = df['Summary'].dropna().tolist()
    return phrases

def find_themes(phrases):
    # Load the pre-trained sentence transformer model
    #model = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed the phrases
    #embeddings = model.encode(phrases, show_progress_bar=True)

    # The main representation of a topic
    main_representation = KeyBERTInspired()

    # Additional ways of representing a topic
    aspect_model1 = PartOfSpeech("en_core_web_sm")
    aspect_model2 = [KeyBERTInspired(top_n_words=30), MaximalMarginalRelevance(diversity=.5)]

    # Add all models together to be run in a single `fit`
    representation_model = {
        "Main": main_representation,
        "Aspect1": aspect_model1,
        "Aspect2": aspect_model2
    }

    topic_model = BERTopic(representation_model=representation_model)
    topics, probs = topic_model.fit_transform(phrases)

    return topic_model, topics, probs

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_csv> <output_file>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_file = sys.argv[2]

    phrases = load_data(csv_path)
    topic_model, topics, probs = find_themes(phrases)

    # Get topic info
    topic_info = topic_model.get_topic_info()

    # Print the topic info
    print(topic_info)
    print(f"Topic information written to {output_file}")

    topic_info.to_csv(output_file, index=False)

    # Get the topics and their representative words
    for topic in topic_info.head(10).Topic:
        if topic == -1:
            continue
        print(f"Topic {topic}:")
        print(topic_model.get_topic(topic))
        print("\n")

if __name__ == "__main__":
    main()
