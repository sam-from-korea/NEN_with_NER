import spacy
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import json
import re
import pandas as pd
import spacy
from spacy.tokens import Doc, Span

def notNaN(num):
    return num == num

def isNaN(num):
    return num != num

def creatlabelTerm(label, lTerm):
    doc = pd.read_csv("../database/corpus/labels/"+ label + ".csv", low_memory=False)
    f = open("../database/Label2Train/labels.jsonl","a")
    i=0
    for var in doc["variation"]:
        if(notNaN(doc["RC"][i]) and doc["RC"][i]!="xxx"):
            lTerm.append(doc["RC"][i])
        elif(isNaN(doc["RC"][i])):
            lTerm.append(var)
        i+=1

input_file = './../database/corpus/preparation/data125_900.jsonl'
conll_data = []
# Create a blank Spacy model
nlp = spacy.blank('en')
docs = []  # This will hold the Doc objects
model_name = "./../database/ablation/output/2/"  
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

ent = ""
lab = ""

with open("./../database/ablation/2/listvar2.txt", 'r') as f:
    lTerm = [line.strip() for line in f]
with open("./../database/ablation/2/listDelete2.txt", 'r') as f:
    ldel = [line.strip() for line in f]

flag = 0
listNewTerm = []
lab = conll_data[0][1][2:]

for i in range(len(conll_data)):
    if(conll_data[i][1][:1]=="B"):
        if(ent not in lTerm and ent!="[SEP]"):
            listNewTerm.append([ent, lab])
            print(f"{ent}: {lab}")
        ent = conll_data[i][0]
    else:
        if(conll_data[i][0]=="-" or ent[-1]=="-"):
            ent = ent + conll_data[i][0]
        else:
            ent = ent + " " + conll_data[i][0]
    lab = conll_data[i][1][2:]
listNewTerm = list(set(tuple(sublist) for sublist in listNewTerm))
listNewTerm = [list(sublist) for sublist in listNewTerm]


# 创建 pandas DataFrame
df = pd.DataFrame(listNewTerm, columns=["Entity", "Label"])

lrf = []

for i in range(len(df["Entity"])):
    ent = df["Entity"][i]
    if df["Entity"][i] in ldel:
        lab = df["Label"][i]
        lrf.append([ent,lab])
dfrefound = pd.DataFrame(lrf, columns=["Entity", "Label"])
dfrefound.to_csv("./../database/ablation/2/refound.csv", index=False)
    
# 将 DataFrame 写入 CSV 文件
df.to_csv("./../database/ablation/2/datanewterms.csv", index=False)