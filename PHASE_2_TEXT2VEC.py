# The below code computes the text embeddings

# importing the required libraries
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.notebook import tqdm
# Load pre-trained BERT model and tokenizer
model_name = 'roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

sentences = ['cable_spool_fort', 'easy_money', 'lw1', 'the_black_willow'] # story names
total_embeddings = {}
# Tokenize and convert to tensor
for sentence in tqdm(sentences):

  with open(f'/content/drive/MyDrive/BV/{sentence}.txt', 'r') as file:
    file_contents = file.read()
  inputs = tokenizer(file_contents, return_tensors='pt', padding=True, truncation=True)
  file_embedding = {}
  # Get model outputs
  with torch.no_grad():
      outputs = model(**inputs)

  # Extract embeddings from the model's hidden states
  embeddings = outputs.hidden_states[-1]  # Last layer's hidden states

  # Get the word embeddings for each token
  word_embeddings = embeddings.mean(dim=1)  # Average pooling of token embeddings

  # Print the word embeddings
  for i, token in enumerate(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])):
      file_embedding[token] = word_embeddings[0][i]
  total_embeddings[sentence] = file_embedding
total_embeddings

import pickle
# open the text embeddings file
with open('/content/drive/MyDrive/BV/text_embeddings.pkl', 'wb') as f:
  pickle.dump(total_embeddings, f)

f = {}

# Iterate through total_embeddings dictionary
for key, values in total_embeddings.items():
    l = []
    for k, v in values.items():
        l.append(v)
    # Append tensors with value 0.0 to make the list 512 elements long
    while len(l) < 512:
        l.append(torch.tensor(0.0))
    f[key] = l

print(f)

import pickle
#saving the final embeddings
with open('/content/drive/MyDrive/BV/text_embeddings_final.pkl', 'wb') as ff:
  pickle.dump(f, ff)