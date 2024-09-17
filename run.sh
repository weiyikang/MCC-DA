# Digit5 dataset
nohup python main.py --config DigitFive.yaml --target-domain mnist -bp ../../../ --temperature 0.8 --tic 1.0 --sic 0.01 --pl 3 --pj 0 --tau 0.05 --seed 1 --gpu 0  > ./log/digit5_ac_tau005_tic1_scic001_mnist_seed1.txt 2>&1 &

nohup python main.py --config DigitFive.yaml --target-domain mnistm -bp ../../../ --temperature 0.8 --tic 1.0 --sic 0.01 --pl 3 --pj 0 --tau 0.05 --seed 1 --gpu 1  > ./log/digit5_ac_tau005_tic1_scic001_mnistm_seed1.txt 2>&1 &

nohup python main.py --config DigitFive.yaml --target-domain svhn -bp ../../../ --temperature 0.8 --tic 1.0 --sic 0.01 --pl 3 --pj 0 --tau 0.05 --seed 1 --gpu 2  > ./log/digit5_ac_tau005_tic1_scic001_svhn_seed1.txt 2>&1 &

nohup python main.py --config DigitFive.yaml --target-domain syn -bp ../../../ --temperature 0.8 --tic 1.0 --sic 0.01 --pl 3 --pj 0 --tau 0.05 --seed 1 --gpu 3  > ./log/digit5_ac_tau005_tic1_scic001_syn_seed1.txt 2>&1 &

nohup python main.py --config DigitFive.yaml --target-domain usps -bp ../../../ --temperature 0.8 --tic 1.0 --sic 0.01 --pl 3 --pj 0 --tau 0.05 --seed 1 --gpu 4  > ./log/digit5_ac_tau005_tic1_scic001_usps_seed1.txt 2>&1 &
