# Mapping Language to Code in Programmatic Context

### Install requirements

Pytorch 0.3 (minor changes needed for 0.4)
```
pip install antlr4-python3-runtime==4.6
pip install allennlp==0.3.0
pip install ipython
```

### Download data from Google drive
```
mkdir concode
cd concode
```
Download data from: https://drive.google.com/drive/folders/1kC6fe7JgOmEHhVFaXjzOmKeatTJy1I1W into this folder.

### Create production rules. This restricts the data to 100000 train and 2000 valid/test. If your resources can support it,  you can use more.
`python build.py -train_file concode/train_shuffled_with_path_and_id_concode.json -valid_file concode/valid_shuffled_with_path_and_id_concode.json -test_file concode/test_shuffled_with_path_and_id_concode.json  -output_folder data  -train_num 100000 -valid_num 2000`

### Prepare pytorch datasets
```
mkdir data/d_100k_762
python preprocess.py -train data/train.dataset -valid data/valid.dataset -save_data data/d_100k_762/concode -train_max 100000 -valid_max 2000
``` 

### Train seq2seq
`python train.py -dropout 0.5  -data data/d_100k_762/concode -save_model data/d_100k_762/s2s -epochs 30 -learning_rate 0.001 -seed 1123 -enc_layers 2 -dec_layers 2  -batch_size 50 -src_word_vec_size 1024 -tgt_word_vec_size 512 -rnn_size 1024 -encoder_type regular -decoder_type regular -copy_attn `

### Train seq2prod
`python train.py -dropout 0.5  -data data/d_100k_762/concode -save_model data/d_100k_762/s2p -epochs 30 -learning_rate 0.001 -seed 1123 -enc_layers 2 -dec_layers 2  -batch_size 20 -src_word_vec_size 1024 -tgt_word_vec_size 512 -rnn_size 1024 -encoder_type regular -decoder_type prod -brnn -copy_attn `

### Train Concode
`python train.py -dropout 0.5  -data data/d_100k_762/concode -save_model data/d_100k_762/concode/ -epochs 30 -learning_rate 0.001 -seed 1123 -enc_layers 2 -dec_layers 2  -batch_size 20 -src_word_vec_size 512 -tgt_word_vec_size 512 -rnn_size 512 -decoder_rnn_size 1024 -encoder_type concode -decoder_type concode -brnn -copy_attn -twostep -method_names -var_names`

Prediction:

On Dev: 
```
ipython predict.ipy -- -start 5 -end 30 -beam 3 -models_dir  data/d_100k_762/concode/ -test_file data/valid.dataset -tgt_len 500 
```
On Test (Use best epoch from dev): 
```
ipython predict.ipy -- -start 15 -end 15 -beam 3 -models_dir  data/d_100k_762/concode/ -test_file data/test.dataset -tgt_len 500 
```

For other model types, use the appropriate `-models_dir`.

