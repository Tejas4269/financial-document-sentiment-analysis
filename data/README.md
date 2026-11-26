# Dataset setup

This project uses the **Financial PhraseBank** dataset described by Malo et al. (2014).

The corpus contains financial news sentences labelled positive, neutral, or negative. For this binary classification task, neutral examples are excluded:
- positive → 1
- negative → 0
- neutral → excluded

## Obtain the data

1. Find the Financial PhraseBank dataset through the authors' project/research page or a trusted academic dataset repository.
2. Review the dataset's licence and terms before downloading or using it.
3. Choose an agreement-level file, preferably `Sentences_75Agree.txt`.
4. Place it at `data/raw/Sentences_75Agree.txt`.

The dataset is not included in this repository. The code expects each line in the standard format: sentence text, followed by `@`, followed by the sentiment label. For example: `The company reported higher earnings@positive`.
