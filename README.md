# Impediment Topic Modeling Tool

This tool performs topic modeling on a set of textual data loaded from a CSV file, using advanced natural language processing techniques. It is designed to classify and group impediments into distinct topics, providing insights into the underlying themes.

## Features

- **Topic Modeling**: Uses BERTopic to identify and classify topics within the data.
- **Advanced NLP Techniques**: Utilizes SentenceTransformer for embedding sentences and SpaCy for part-of-speech tagging.
- **Comprehensive Output**: Outputs topic information to a CSV file for easy review and further analysis.

## Requirements

- Python 3.6+
- `pandas`
- `nltk`
- `scikit-learn`
- `transformers`
- `sentence-transformers`
- `bertopic`
- `spacy`

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/impediment-topic-modeling-tool.git
   cd impediment-topic-modeling-tool
   ```

2. **Create a virtual environment and activate it:**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Download the SpaCy model:**

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. **Prepare your CSV file:**
   Ensure your CSV file has a column named `Summary` containing the text data to be analyzed.

2. **Run the tool:**

   ```bash
   python main.py <path_to_csv> <output_file>
   ```

   Replace `<path_to_csv>` with the path to your input CSV file and `<output_file>` with the desired path for the output CSV file.

## Example

```bash
python main.py data/impediments.csv output/topics.csv
```

## Output

The tool generates an output CSV file containing the following columns:
- **Topic**: The topic identifier.
- **Count**: The number of documents in this topic.
- **Name**: A short name for the topic.
- **Representation**: The main representation of the topic.
- **Aspect1**: Additional representation using part-of-speech tagging.
- **Aspect2**: Additional representation using KeyBERTInspired and MaximalMarginalRelevance.
- **Representative_Docs**: Examples of documents classified under this topic.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

Feel free to reach out if you have any questions or need further assistance. Happy topic modeling!

