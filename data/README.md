# Dataset setup

This project is designed for the **Financial PhraseBank** dataset by Malo et al. (2014).

The dataset contains financial news sentences labelled as positive, neutral, or negative by human annotators. For this binary classification project, neutral examples are excluded:
- positive → 1
- negative → 0
- neutral → excluded

## Obtain the data

1. Visit the dataset page: https://www.researchgate.net/publication/251231 વિચાર
2. Obtain the Financial PhraseBank dataset under its applicable research/use terms.
3. Select one of the agreement-level sentence files, preferably `Sentences_75Agree.txt`.
4. Place the text file at `data/raw/Sentences_75Agree.txt`.

The dataset is not included in this repository. Please review and comply with the dataset's licence and terms before using or redistributing it.

The training script accepts the standard Financial PhraseBank format, where each line contains a sentence followed by `@` and its sentiment label.
