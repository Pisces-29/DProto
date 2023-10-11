# Density-aware Prototypical network(D-Proto)
Codes for our EMNLP 2023 paper: Density-Aware Prototypical Network for Few-Shot Relation Classification.

## Environments

- `python 3.6.2`
- `torch 1.7.1`
- `transformers 4.5.1`
- `scikit-learn 0.24.2`

## Datasets

### FewRel

You can find the training and validation data here: [FewRel](https://thunlp.github.io/2/fewrel2_nota.html).

## Training

### O-Proto model: 

``` bash 
python train.py --dataset fewrel --N 5 --K 5 --Q 3 --batch_size 2 --model oproto --encoder bert --max_length 128 --trainNA 0.5 --optim adamw --hidden_size 768 --seed {}
```

### Bert-Pair model: 

``` bash 
python train.py --dataset fewrel --N 5 --K 5 --Q 3 --batch_size 2 --model pair --encoder bert --max_length 128 --trainNA 0.5 --optim adamw --hidden_size 768 --seed {} --pair
```

### MNAV model: 

``` bash 
python train.py --dataset fewrel --N 5 --K 5 --Q 3 --batch_size 2 --model mnav --vector_num 20 --encoder bert --max_length 128 --trainNA 0.5 --optim adamw --hidden_size 768 --seed {}
```

### D-Proto model: 

``` bash 
python train.py --dataset fewrel --N 5 --K 5 --Q 3 --batch_size 2 --model dproto --gamma 1e-5 --threshold 0.9 --encoder bert --max_length 128 --trainNA 0.5 --optim adamw --hidden_size 768 --seed {}
```

* `N`: N in N-way K-shot.
* `K`: K in N-way K-shot.
* `Q`: Sample Q query instances in the query set.
* `trainNA`: NOTA rate in training phase. In testing, test results under 0.15, 0.3, 0.5 NOTA rates are obtained respectively.
* `seed`: seed. 5/10/15/20/25.
