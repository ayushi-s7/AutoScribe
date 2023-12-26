from flask import Flask, render_template, request, jsonify
import Levenshtein
from auto_complete import autocomplete  # Import the autocomplete function from autocomplete.py
from load_model import summarize

app = Flask(__name__)

# Load your dictionary of correct words from a file
def load_dictionary(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file]

# Suggest corrections for a misspelled word
def suggest_corrections(dictionary, user_input):
    nearest_word = None
    nearest_distance = float('inf')

    for word in dictionary:
        distance = Levenshtein.distance(user_input, word)
        if distance < nearest_distance:
            nearest_word = word
            nearest_distance = distance

    return nearest_word

# Load your dictionary from a file
dictionary = load_dictionary('words.txt')

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/summarize_text', methods=['POST'])
def summarize_text():
    data = request.get_json()
    input_text = data['input_text']
    summarized_text = summarize(input_text)  # Call the summarize function

    return jsonify({"summary": summarized_text})

# Add a new route to handle word suggestions
@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    misspelled_word = request.json['word']
    suggested_word = suggest_corrections(dictionary, misspelled_word)
    return jsonify({"suggested_word": suggested_word})

# Add a route for autocomplete
@app.route('/autocomplete_word', methods=['POST'])
def autocomplete_word():
    data = request.get_json()
    input_sentence = data['sentence']
    next_word = autocomplete(input_sentence)  # Call the autocomplete function
    return jsonify({"next_word": next_word})

if __name__ == '__main__':
    app.run(debug=True)