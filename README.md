# NLP-Primer

This repository contains the implementation of [Primer-EZ](https://arxiv.org/abs/2109.08668). The network architecture will be added in at a later stage.

## Model Training and Inference
Before training the model, process the [Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset by running
```
python process_movie_dialogue_subword.py
```
command, followed by
```
python train_movie_dialogue_sw_gpt_primer.py
```
to train the model. After training is complete, run
```
python infer_movie_dialogue_sw_gpt_primer.py
```
to perform inference using the model.
