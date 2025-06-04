# %% [markdown]
# Nesta aula, vamos aprender como trabalhar com ngramas e stopwords, utilizando a bilioteca NLTK.

# %% [markdown]
# # NGRAMAS, Stopwords e NLTK
# ### Autor: Lucas Ferro Antunes de Oliveira
# #### HAILab - PPGTS - PUCPR
# ### Adaptado por: Vitor Hugo Dias Santos
# #### Sistemas de Informação - UFMG
# 
# lucas.ferro.2000@hotmail.com
# 
# vhugosantos@gmail.com

# %% [markdown]
# #Tokenização e normalização do corpus

# %%
# prompt: converta um arquivo pdf em txt padrão utf-8

%pip install PyPDF2

import PyPDF2
import re

def convert_pdf_to_txt(pdf_path, txt_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Remove extra whitespace and newlines
            text = re.sub(r'\s+', ' ', text)

            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

        print(f"PDF '{pdf_path}' converted to '{txt_path}' successfully.")
    except FileNotFoundError:
        print(f"Error: File '{pdf_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
pdf_file_path = 'PDF/Isaac Asimov - Eu, Robô.pdf'  # Replace with your PDF file path
txt_file_path = 'output.txt'      # Replace with desired output file path

convert_pdf_to_txt(pdf_file_path, txt_file_path)


# %%
# Instalação do NLTK
%pip install nltk==3.6.2

# %%
%pip install pandas nltk matplotlib wordcloud Pillow


# %%
# Importação de bibliotecas
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from datetime import datetime
from collections import Counter
from nltk import ngrams
import string
from wordcloud import WordCloud,STOPWORDS,ImageColorGenerator
import matplotlib.pyplot as plt
from PIL import Image
import re
nltk.download('punkt')

# %%
# pd.set_option('max_columns', None)
# pd.set_option('max_colwidth', None)

# %%
# Pega todas as pontuações
remove_pt = string.punctuation
remove_pt + "•"
remove_pt

# %%
# Baixa as stopwords para o português no NLTK
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words_pt = set(stopwords.words('portuguese'))
len(stop_words_pt)
stop_words_pt

# %%
stop_words_pt.add('ser')
stop_words_pt.add('entao')
stop_words_pt.add('de')

stop_words_pt

# %%
stop_words_pt.add('ainda')
stop_words_pt.add('porém')
stop_words_pt

# %%
PATH = 'output.txt'

# %%
with open(PATH , 'r', encoding='utf8') as f:
    filecontent = f.read()

print(filecontent[0:])

# %%
type(filecontent)

# %%
len(filecontent) # número de tokens

# %% [markdown]
# ## Transformando o texto completo em sentenças (tokenizer do NLTK)

# %%
sentencas = []
for sentence in sent_tokenize(filecontent, language = 'portuguese'):
    sentencas.append(sentence)

# %%
index = 1
for sentenca in sentencas[0:100]: # mostrando as 100 primeiras
    print(f'{index}: {sentenca}')
    index+=1

# %% [markdown]
# ## Segmentação por quebra de linha e depois pelo tokenizer do NLTK

# %%
sentencas_linha = []
for sentence in filecontent.split('\n'):
    if sentence != '':
        for processed_sentence in sent_tokenize(sentence, language = 'portuguese'):
            sentencas_linha.append(processed_sentence)

# %%
sentencas_linha[0:10]

# %%
index = 1
for sentenca in sentencas_linha[0:100]: # mostrando as 100 primeiras
    print(f'{index}: {sentenca}')
    index+=1

# %% [markdown]
# ## Tokenização de cada sentença em palavras (tokenizer do NLTK)
# 
# 
# 
# 

# %%
sentencas_tokenizadas = []

for sentenca in sentencas_linha:
    tokenized_sentence = word_tokenize(sentenca, language='portuguese')
    sentencas_tokenizadas.append(tokenized_sentence)
index = 1
for tokens in sentencas_tokenizadas[0:100]: # mostrando as 100 primeiras
    print(f'{index}: {tokens}')
    index+=1

# %% [markdown]
# ## Pre-processamento dos elementos tokenizados
# A ideia aqui é retirar todas as palavras que pertencem a lista de stopwords, deixar tudo em minúsculos, retirar espaços e quebras de linhas adicionais desnecessários.

# %%
from typing import TextIO
sent_tokenizada_preprocessed = []
for sent_tokenizada in sentencas_tokenizadas:
    raw = [token.lower() for token in sent_tokenizada]

    raw = [''.join(c for c in s if c not in remove_pt+'–'+'🙁'+'\’'+'\”'+"“"+"•") for s in raw]
    raw = [re.sub(r"\d+[.,]?\d*","", s) for s in raw]
    raw = [s for s in raw if s not in stop_words_pt] # stopwords
    raw = [' '.join(s.split()) for s in raw if s]
    string = ' '.join(raw).rstrip().lstrip()
    if string != '':
        sent_tokenizada_preprocessed.append(string)

index = 1
for texto in sent_tokenizada_preprocessed[0:100]: # mostrando as 100 primeiras
    print(f'{index}: {texto}')
    index+=1


# %% [markdown]
# #NGramas

# %%
len(sent_tokenizada_preprocessed)

# %%
import os
ngram_value = 1 #muda o tamanho do engrama
most_common_value = 100

ngram_counts = [list(ngrams(s.split(), ngram_value)) for s in sent_tokenizada_preprocessed]
flat_ngram_counts = [item for sublist in ngram_counts for item in sublist]
ngram_list = Counter(flat_ngram_counts)

common = ngram_list.most_common(most_common_value)

df_common = pd.DataFrame(common, columns = ['Ngram','Count'])
index = 1
for n_gram in ngram_counts[0:100]: # mostrando as 100 primeiras
    print(f'{index}: {n_gram}')
    index+=1

# %%
index = 1
for n_gram in flat_ngram_counts[0:100]: # mostrando as 100 primeiras
    print(f'{index}: {n_gram}')
    index+=1

# %%
len(ngram_list)

# %%
common

# %%
df_common.head(30)

# %%
# Quantidade de palavras
len(flat_ngram_counts)

# %%
# Quantidade de palavras únicas
len(ngram_list)

# %%
color = 'black'
height = 400
width = 800
max_words = 2000
colormap = 'rainbow'
size_X = 50
size_Y = 50

str_text=" ".join(sent_tokenizada_preprocessed)


wordcloud = WordCloud(background_color = color, max_words = max_words, max_font_size = 120, colormap = colormap, height = height, width = width).generate(str_text)

X = size_X/2.54
Y = size_Y/2.25

fig = plt.figure(figsize = [X, Y])
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.box(False)
plt.show()

# %%
%pip install PyPDF2


# %% [markdown]
# fazer a frequencia, Carlos

# %%
# prompt: leia um arquivo pdf e gere um txt com utf8

import PyPDF2
import re

def convert_pdf_to_txt(pdf_path, txt_path):
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()

            # Remove extra whitespace and newlines
            text = re.sub(r'\s+', ' ', text)

            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)

        print(f"PDF '{pdf_path}' converted to '{txt_path}' successfully.")
    except FileNotFoundError:
        print(f"Error: File '{pdf_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage (replace with your file paths):
pdf_file_path = '/content/DalleScienzeDelLinguaggioAllEducazionePlurilingue.ocr.pdf'
txt_file_path = '/content/DalleScienzeDelLinguaggioAllEducazionePlurilingue.txt'

convert_pdf_to_txt(pdf_file_path, txt_file_path)


# %%
# prompt: coloque cada página do pdf num arquivo txt assinumeroado pagina001.txt, pagina002.txt etc. dentro de um diretorio.

import os
import PyPDF2

def split_pdf_to_txt_files(pdf_path, output_dir):
    """Splits a PDF file into individual text files, one per page."""

    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Format the filename with leading zeros
                filename = os.path.join(output_dir, f"pagina{page_num + 1:03d}.txt")

                with open(filename, 'w', encoding='utf-8') as txt_file:
                    txt_file.write(text)

            print(f"PDF '{pdf_path}' split into text files in '{output_dir}' successfully.")
    except FileNotFoundError:
        print(f"Error: File '{pdf_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


# Example usage:
pdf_file_path = '/content/DalleScienzeDelLinguaggioAllEducazionePlurilingue.ocr.pdf'
output_directory = '/content/pdf_pages'  # Directory to save the text files
split_pdf_to_txt_files(pdf_file_path, output_directory)


# %%
!tar -czvf /content/pdf_pages.tgz /content/pdf_pages/

