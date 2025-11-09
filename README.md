# Legendagem de Imagens com Deep Learning: Comparativo de Decoders

Este repositório documenta o desenvolvimento de um projeto de *Image Captioning* (legendagem automática de imagens) que compara diferentes arquiteturas de decoders (LSTM, GRU e Transformer) com um encoder CNN pré-treinado (ResNet50). O objetivo é construir um pipeline completo em PyTorch, desde o pré-processamento de dados até o treinamento e a avaliação comparativa dos modelos.

O projeto foi desenvolvido como parte da disciplina de Tópicos Especiais em Matemática Aplicada da Universidade de Brasília (UnB/FCTE), ministrada pelo Professor Vinicius Rispoli no semestre 2025.2.

## Contribuidores

<div align="center">

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com)

[`@matheuslemesam`](https://github.com/matheuslemesam)

</div>

## Visão Geral do Projeto

O notebook `ImageCaptioningDL.ipynb` foi estruturado para ser executado no Google Colab, aproveitando GPUs para acelerar o treinamento. O fluxo de trabalho pode ser dividido nas seguintes etapas:

### 1. Preparação do Ambiente e Dados
- **Ambiente Colab:** O ambiente é configurado para se conectar ao Google Drive e copiar o dataset (Flickr8k) para o armazenamento local do Colab, acelerando o acesso aos dados.
- **Dependências:** Instalação de bibliotecas essenciais como `PyTorch`, `spaCy` (para tokenização de legendas) e `torchmetrics` (para avaliação com a métrica BLEU).
- **Pré-processamento:**
    - **Vocabulário:** Uma classe `Vocabulary` é construída para mapear cada palavra das legendas de treino a um índice numérico, tratando palavras raras com um token `<unk>`.
    - **Dataset e DataLoader:** A classe `FlickrDataset` carrega imagens e suas legendas correspondentes. Um `DataLoader` customizado com `collate_fn` aplica *padding* às legendas para criar batches de tamanho uniforme.
    - **Aumento de Dados (Data Augmentation):** Transformações como `RandomHorizontalFlip` e `RandomCrop` são aplicadas às imagens de treino para aumentar a robustez do modelo.

### 2. Arquitetura Encoder-Decoder

O projeto utiliza uma arquitetura Encoder-Decoder, onde uma CNN extrai as características da imagem e um modelo sequencial gera a legenda.

- **Encoder (CNN):**
  - Uma **ResNet-50**, pré-treinada no dataset ImageNet, atua como o encoder.
  - A camada de classificação final da ResNet-50 é removida e substituída por uma camada linear que mapeia as características visuais para um vetor de embedding. Este vetor serve como o contexto inicial para os decoders.

- **Decoders (RNNs e Transformer):**
  - O projeto implementa e compara três arquiteturas de decoder diferentes para gerar as legendas palavra por palavra:
    1. **DecoderLSTM:** Utiliza uma rede Long Short-Term Memory (LSTM), uma arquitetura clássica para processamento de sequências.
    2. **DecoderGRU:** Utiliza uma Gated Recurrent Unit (GRU), uma variação mais simples e computacionalmente mais eficiente que a LSTM.
    3. **DecoderTransformer:** Utiliza a arquitetura Transformer, baseada em mecanismos de auto-atenção, que se tornou o padrão em muitas tarefas de processamento de linguagem natural.

### 3. Estratégia de Treinamento

Para garantir uma comparação justa, foi adotada uma estratégia de treinamento sequencial:
1. **Treinamento Inicial:** O **Encoder** e o **DecoderLSTM** são treinados juntos. Ao final, os pesos do encoder treinado são salvos.
2. **Treinamento com Encoder Congelado:** Para os outros dois modelos, o encoder treinado é carregado e seus pesos são "congelados" (*frozen*). Apenas os pesos dos decoders **GRU** e **Transformer** são atualizados.
- **Otimização:** O otimizador Adam é utilizado com um scheduler `ReduceLROnPlateau`, que reduz a taxa de aprendizado se a perda de validação não melhorar.
- **Early Stopping:** O treinamento é interrompido se a perda de validação não apresentar melhora após um número definido de épocas, evitando overfitting.

### 4. Avaliação e Benchmark

- **Métrica BLEU:** O desempenho de cada modelo é avaliado quantitativamente usando a métrica BLEU (Bilingual Evaluation Understudy), que mede a sobreposição de n-gramas entre as legendas geradas e as de referência. Foram calculados os scores BLEU-1, BLEU-2, BLEU-3 e BLEU-4.
- **Análise Qualitativa:** Além das métricas, o projeto inclui uma análise visual onde legendas geradas pelos três modelos são comparadas lado a lado com as legendas reais para imagens aleatórias do conjunto de validação.

### 5. Resultados e Benchmark Final

Após o treinamento e a avaliação dos três decoders, os resultados foram consolidados para uma análise comparativa.

| Decoder | BLEU-1 (%) | BLEU-2 (%) | BLEU-3 (%) | BLEU-4 (%) |
| :--- | :--- | :--- | :--- | :--- |
| **GRU** | **59.4** | **38.7** | **25.3** | **16.4** |
| **LSTM** | 58.4 | 37.9 | 24.6 | 15.9 |
| **Transformer** | 57.9 | 36.9 | 24.0 | 15.5 |

**Conclusão do Benchmark:**

A abordagem com **GRU** alcançou o melhor desempenho em todas as métricas BLEU, superando ligeiramente a LSTM e o Transformer. Embora o Transformer seja o estado da arte em muitas tarefas de NLP, a GRU, com sua arquitetura mais simples, provou ser altamente eficaz para este dataset e configuração específicos. A LSTM apresentou um desempenho muito próximo, enquanto o Transformer, que geralmente requer mais dados e ajuste fino de hiperparâmetros, ficou em terceiro lugar. Isso sugere que para datasets de tamanho moderado como o Flickr8k, arquiteturas RNN bem ajustadas ainda são extremamente competitivas.


