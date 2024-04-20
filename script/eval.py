import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json
import re
import pandas as pd
import spacy
from spacy.tokens import Doc, Span
input_file = './../database/corpus/preparation/merged.jsonl'
conll_data = []
# Create a blank Spacy model
nlp = spacy.blank('en')
docs = []  # This will hold the Doc objects
model_name = "./../output/nano/"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

with open(input_file, 'r') as f:
    data = [json.loads(line) for line in f]
# For each example in your data
for example in data:
    text = example['text']
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)

    predicted_label_ids = predictions[0].numpy()
    predicted_labels = [model.config.id2label[label_id] for label_id in predicted_label_ids]
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    combined_labels = []
    combined_tokens = []
    entity_spans = []

    for token, label in zip(tokens, predicted_labels):
        if token.startswith("##"):
            combined_tokens[-1] = combined_tokens[-1] + token[2:]
        else:
            combined_tokens.append(token)
            combined_labels.append(label)
    # Create a doc with the tokens and spaces
    doc = Doc(nlp.vocab, words=combined_tokens, spaces=[True]*len(combined_tokens))

    # Remove '##' from the text
    #doc_text = ' '.join([token.text.replace(' ##', '') for token in doc])

    entity_spans = []
    for i, label in enumerate(combined_labels):
        l1 = []
        if label != "O":
            span = Span(doc, i, i+1, label)
            entity_spans.append(span)
            l1.append(combined_tokens[i])
            l1.append(label)
            conll_data.append(l1)
    # Assign the entities to the doc
    doc.ents = entity_spans
    docs.append(doc)

from spacy import displacy
#displacy.serve(docs, style='ent')
html = displacy.render(docs, style='ent', page=True)
with open('output.html', 'w', encoding='utf-8') as f:
    f.write(html)