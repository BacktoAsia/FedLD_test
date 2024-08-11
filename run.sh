nohup python -u federated_main.py --train_rule FedAvg --dataset synthetic_0.5_0.5 --local_epoch 1 --lr 0.1 --local_bs 8 --device cuda:0 > out/fedavg_synthetic_0.5_0.5_lr_0.1.out &
nohup python -u federated_main.py --train_rule FedLD --dataset synthetic_1_1 --margin_loss_penalty 0.1 --local_epoch 1 --lr 0.1 --local_bs 8 --device cuda:0 > out/fedld_wo_P_lambda_0.1_synthetic_1_1_lr_0.1.out &

nohup python -u federated_main.py --train_rule FedAvg --dataset cifar --iid 1 --local_bs 32 --local_epoch 5 --lr 0.01 --num_users 20 --frac 1.0 --device cuda:1 > out/fedavg_cifar_iid.out &
nohup python -u federated_main.py --train_rule FedLD --margin_loss_penalty 0.03 --dataset cifar --iid 1 --local_bs 32 --local_epoch 5 --lr 0.01 --num_users 20 --frac 1.0 --device cuda:2 > out/fedld_lambda_0.03_cifar_iid.out &

nohup python -u federated_main.py --train_rule FedAvg --dataset cifar --iid 0 --noniid_s 20 --local_bs 32 --local_epoch 5 --lr 0.1 --num_users 20 --frac 1.0 --device cuda:0 > out/fedavg_cifar_noniid_20_lr_0.1.out &
nohup python -u federated_main.py --train_rule FedLD --margin_loss_penalty 0.1 --dataset cifar --iid 0 --noniid_s 20 --local_bs 32 --local_epoch 5 --lr 0.001 --num_users 20 --frac 1.0 --device cuda:0 > out/fedld_lambda_0.1_cifar_noniid_20_lr_0.001.out &

nohup python -u federated_main.py --train_rule FedAvg --dataset retina --retina_split 3 --num_users 5 --local_bs 32 --lr 0.01 --epochs 200 --local_epoch 1 --device cuda:1 > out/fedavg_retina_split3.out &
nohup python -u federated_main.py --train_rule FedLD --dataset retina --retina_split 3 --num_users 5 --local_bs 32 --lr 0.01 --epochs 200 --local_epoch 1 --margin_loss_penalty 0.03 --k_proportion 0.8 --device cuda:0 > out/fedld_retina_split3.out &
