# Word RNN for Sentence Completion
A pytorch implementation of the word-level recurrent neural network for sentence completion.
The code is based on [Word-level language modeling RNN](https://github.com/pytorch/examples/tree/master/word_language_model), and importance sampling module is from [PyTorch Large-Scale Language Model](https://github.com/rdspring1/PyTorch_GBW_LM).

## Requirements
- torchvision >= 0.2.0
- torch >= 0.3.0.post4
- numpy >= 1.13.3
- pandas >= 0.21.0
- nltk >= 3.2.5
- tqdm >= 4.19.5
- Cython >= 0.27.3
> pip3 install -r requirements.txt

## Setup
- Build Log_Uniform Sampler according to [Link](https://github.com/rdspring1/PyTorch_GBW_LM).
- Download `punkt` package in `nltk`.

## Datasets
- Microsoft Research Sentence Completion Challenge -
    Training and Test dataset can be downloaded from [Link](https://drive.google.com/open?id=0B5eGOMdyHn2mWDYtQzlQeGNKa2s). Store the downloaded test data in `./data/completion/`.
- Scholastic Aptitude Test sentence completion questions -
    Collected questions are provided in `./data/completion/SAT_set_filled.csv`.
- Nineteenth century novels (19C novels) -
    Extract `./data/prepro/guten.tgz` of preprocessed files.
- One Billion Word Benchmark (1B word) - [Link](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz)

## Run
### Training
> python3 train.py --cuda --save_dir mynet

Default arguments are set for training with 19C novels. Argument settings for training with the 1B word benchmark are presented in the following table.

| Argument		| 19C novels	| 1B word	|
|---------------|---------------|-----------|
| corpus		| guten			| gbw		|
| emsize		| 200			| 500		|
| nhid			| 600			| 2000		|
| outsize		| 400			| 500		|
| lr			| 0.5			| 1.0		|
| decay_after	| 5				| 1			|
| decay_rate	| 0.5			| 0.8		|
| batch_size	| 20			| 100		|
| nsampled		| -1			| 8192		|

### Sentence completion
> python3 sent_cmplt.py --cuda --save_dir mynet

## Results
| corpus	| bidirec	| MSR accuracy	| SAT accuracy	|
|:----------|:----------|:-------------:|:-------------:|
| guten     | False		| 69.4 (0.8)*	| 29.6 (1.5)*	|
| guten		| True		| 72.3 (1.1)*	| 33.3 (2.0)*	|
| gbw		| False		| 63.2			| 66.5			|
| gbw		| True		| 64.1			| 69.1			|

*The mean accuracy of five networks trained with different random initializations is shown with the standard deviation in parentheses.
