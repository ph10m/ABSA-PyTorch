python train.py --model_name lstm --dataset restaurant --dropout 0.2 --l2reg 0.01 --batch_size 3 --learning_rate 1e-3 --num_epoch 20
python train.py --model_name lstm --dataset restaurant_coref_gold_eval --dropout 0.2 --l2reg 0.01 --batch_size 3 --learning_rate 1e-3 --num_epoch 20
python train.py --model_name td_lstm --dataset restaurant --dropout 0.2 --l2reg 0.01 --batch_size 3 --learning_rate 1e-3 --num_epoch 10
python train.py --model_name td_lstm --dataset restaurant_coref_gold_eval --dropout 0.2 --l2reg 0.01 --batch_size 3 --learning_rate 1e-3 --num_epoch 10
python train.py --model_name ram --dataset restaurant --dropout 0.2 --l2reg 0.01 --batch_size 3 --learning_rate 1e-3 --num_epoch 10
python train.py --model_name ram --dataset restaurant_coref_gold_eval --dropout 0.2 --l2reg 0.01 --batch_size 3 --learning_rate 1e-3 --num_epoch 10
python train.py --model_name tnet_lf --dataset restaurant --dropout 0.2 --l2reg 0.01 --batch_size 3 --learning_rate 1e-3 --num_epoch 10
python train.py --model_name tnet_lf --dataset restaurant_coref_gold_eval --dropout 0.2 --l2reg 0.01 --batch_size 3 --learning_rate 1e-3 --num_epoch 10
python train.py --model_name lcf_bert --dataset restaurant --dropout 0.001 --l2reg 0.00001 --batch_size 12 --learning_rate 2e-5 --local_context_focus cdm --SRD 3 --num_epoch 3
python train.py --model_name lcf_bert --dataset restaurant_coref_gold_eval --dropout 0.001 --l2reg 0.00001 --batch_size 12 --learning_rate 2e-5 --local_context_focus cdm --SRD 3 --num_epoch 3
python train.py --model_name lcf_bert --dataset restaurant --dropout 0.001 --l2reg 0.00001 --batch_size 12 --learning_rate 2e-5 --local_context_focus cdw --SRD 6 --num_epoch 3
python train.py --model_name lcf_bert --dataset restaurant_coref_gold_eval --dropout 0.001 --l2reg 0.00001 --batch_size 12 --learning_rate 2e-5 --local_context_focus cdw --SRD 6 --num_epoch 3

