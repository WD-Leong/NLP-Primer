# NLP-Primer

This repository contains the implementation of [Primer-EZ](https://arxiv.org/abs/2109.08668). The network architecture will be added in at a later time.

## Model Training and Inference
Before training the model, process the [Movie Dialogue](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) dataset by running
```
python process_movie_dialogue_subword.py
```
command, followed by
```
python train_movie_dialogue_sw_tf_ver2_primer.py
```
to train the model. After training is complete, run
```
python infer_movie_dialogue_sw_tf_ver2_primer.py
```
to perform inference using the model.

## RMS Normalisation Layer
Following [Gopher](https://www.deepmind.com/blog/language-modelling-at-scale-gopher-ethical-considerations-and-retrieval), `tf_ver2_gpt_rms_primer.py` was added to use RMS Normalisation Layer. No visible difference was noticed apart from a very slight improvement in training wall clock. To use this version, change `import tf_ver2_gpt_primer as tf_gpt` to `import tf_ver2_gpt_rms_primer as tf_gpt` in line 9 of the training main code. Alternatively, `train_movie_dialogue_sw_tf_ver2_rms_primer.py` can be run to train the movie dialogue chatbot model.
