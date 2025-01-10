from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Carregar o tokenizer e o modelo BERT pré-treinado com atenção ativada
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(
    'bert-base-uncased', 
    output_hidden_states=True,    # Se precisar das camadas ocultas
    output_attentions=True        # Ativar a saída de atenções
)

# Configurando o modelo para avaliação
model.eval()

# Definir a pergunta
pergunta = "Quais são as 3 Leis de Newton?"

# Tokenizar a entrada
inputs = tokenizer(pergunta, return_tensors="pt")

# Passar a entrada pelo modelo
with torch.no_grad():
    outputs = model(**inputs)

# Extraindo as atenções
attentions = outputs.attentions

# Verificar se as atenções foram extraídas corretamente
if attentions is None:
    raise ValueError("As atenções não foram retornadas. Verifique se 'output_attentions=True' está configurado.")

# Escolher a camada e a cabeça que desejamos visualizar
camada_selecionada = 0
cabeca_selecionada = 0

# Extrair as atenções da camada e cabeça selecionadas.
# Atenções têm a forma (batch_size, num_heads, sequence_length, sequence_length)
attention_matrix = attentions[camada_selecionada][0, cabeca_selecionada].detach().numpy()

# Obter os tokens (inclui tokens especiais como [CLS] e [SEP])
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Criar um DataFrame para armazenar a matriz de atenção
df = pd.DataFrame(attention_matrix, columns=tokens, index=tokens)

# Criar uma grade para os eixos X e Y
x = np.arange(len(tokens))
y = np.arange(len(tokens))
x, y = np.meshgrid(x, y)

# Obter os valores de atenção para o eixo Z
z = attention_matrix

# Criar o gráfico de superfície
fig = go.Figure(data=[go.Surface(
    x=x,
    y=y,
    z=z,
    colorscale='Viridis',
    colorbar=dict(title='Atenção')
)])

# Personalizar o layout
fig.update_layout(
    title=f'Atenção - Camada {camada_selecionada +1}, Cabeça {cabeca_selecionada +1}',
    scene=dict(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(len(tokens))),
            ticktext=tokens,
            title='Palavra de Entrada'
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(tokens))),
            ticktext=tokens,
            title='Palavra de Foco'
        ),
        zaxis=dict(title='Peso da Atenção')
    )
)

# Exibir o gráfico
fig.show()
