from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from datasets import Dataset, ClassLabel, Sequence

app = Flask(__name__)

# load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dumitrescustefan/bert-base-romanian-uncased-v1")

with open('outputSentence.txt', 'r', encoding="utf8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)


label_names = sorted(set(label for labels in dataset["ner_tags"] for label in labels))
dataset = dataset.cast_column("ner_tags", Sequence(ClassLabel(names=label_names)))

model = AutoModelForTokenClassification.from_pretrained(r"D:\Alex\Licenta\model")


# define a function to predict labels for an input sentence
def predict_labels(input_sentence):
    # encode the input sentence
    tokenized_input = tokenizer.encode(input_sentence, return_tensors='pt')
    # make the prediction and obtain the predicted labels
    with torch.no_grad():
        outputs = model(tokenized_input)
        predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()

    # decode the predicted labels
    predicted_labels = [dataset.features["ner_tags"].feature.names[i] for i in predictions]
    return predicted_labels


def decode_input(input_sentence):
    tokenized_input = tokenizer.encode(input_sentence, return_tensors='pt')
    decoded_input = tokenizer.decode(tokenized_input.squeeze().tolist())
    return tokenizer.tokenize(decoded_input)

def concat_new(input):
    concatenated_text = []
    for line in input:
        newLine = ""
        if "##" in line:
            i = 0
            for charater in line:
                if charater != '#':
                    newLine = newLine + charater
                else:
                    i = i + 1
                    if i==2:
                        i=0
                        newLine = newLine[:-1]
            concatenated_text.append(newLine)
        else:
            concatenated_text.append(line)
    print(concatenated_text)
    return concatenated_text

def print_every(decoded_input , predicted_labels):
    current_word = ""
    current_label = None
    list1 = []
    list2 = []
    for token, label in zip(decoded_input, predicted_labels):
        # if the label has changed, print the current word and its label
        if label != current_label:
            if current_word:
                print(current_word, current_label)
                list1.append(current_word)
                list2.append(current_label)
            current_word = token
            current_label = label
        # if the label is the same, append the current token to the current word
        else:
            current_word += " " + token
    if len(list1)==0 and len(list2)==0:
        list1.append(current_word)
        list2.append(label)
    list_concat = concat_new(list1)

    return list_concat, list2

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # get the input sentence from the form
    input_sentence = request.form['sentence']

    # predict the labels for the input sentence
    predicted_labels = predict_labels(input_sentence)

    decoded_input = decode_input(input_sentence)

    word_list = print_every(decoded_input, predicted_labels)

    print(word_list)
    print(predicted_labels)
    print(decoded_input)
    # display the predicted labels on the webpage
    return render_template('index.html', predicted_labels=predicted_labels, decoded_input = decoded_input, tokenizer=tokenizer, word_list=word_list)

if __name__ == '__main__':
    app.run(debug=True)