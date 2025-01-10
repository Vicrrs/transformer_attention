from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Carregar o tockenizer e o modelo BERT pré-treinado
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Configurando o modelo pra avaliação
model.eval()

# Definir a pergunra
pergunta = "Quais são as 3 Leis de Newton?"

# Tokenizar a entrada
inputs = tokenizer(pergunta, return_tensors="pt")

# Passar a entrada pelo modelo
with torch.no_grad():
    outputs = model(**inputs)

# Extraindo as atenções
attentions = outputs.attentions

# Escolhendo a camada e a cabeca que desejamos visualizar
camada_selecionada = 0
cabeca_selecionada = 0

# Extrair as atenções da camada e acabeça selecionadas.
# Atenções tem a forma (batch_size, num_heads, sequence_length, sequence_length)
attention_matrix = attentions[camada_selecionada][0, cabeca_selecionada].detach().numpy()

# Obter os tokens (inclui tokens especiais como [CLS] e [SEP])
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
