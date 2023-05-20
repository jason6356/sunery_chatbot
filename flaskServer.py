from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
from sklearn.metrics.pairwise import cosine_similarity
import csv

nlp = spacy.load('en_core_web_md')

app = Flask(__name__)
CORS(app)

# Define the route for the chatbot endpoint
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the user's query from the request data
    user_query = request.json['query']

    # TODO: Implement your chatbot logic here to generate a response
    chatbot_response = generate_chatbot_response(user_query)

    # Create a JSON response with the chatbot's response
    response = {
        'response': chatbot_response
    }

    # Return the JSON response
    return jsonify(response)

def read_csv_file(file_path):
    data = {}
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            question = row[0]
            answer = row[1]
            data[question] = answer
    return data

responses = read_csv_file('data.csv')

# Placeholder function to generate chatbot response
def generate_chatbot_response(user_query):
    query_vector = nlp(user_query).vector
    max_similarity = 0
    most_similar_question = ''

    for question in responses:
        question_vector = nlp(question).vector
        similarity = cosine_similarity([query_vector], [question_vector])[0][0]

        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_question = question
        
    return responses.get(most_similar_question)


if __name__ == '__main__':
    app.run(debug=True)
