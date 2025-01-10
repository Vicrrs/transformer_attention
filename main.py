from transformers import BertTokenizer, BertModel
import torch

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
