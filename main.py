from transformers import BertTokenizer, BertModel
import torch

# Carregar o tockenizer e o modelo BERT pré-treinado
tockenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

# Configurando o modelo pra avaliação
model.eval()
