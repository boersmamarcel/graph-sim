python experiments.py -d synthetic_random_100 -i 8 -r 64 --gridsearch --model wwlr --n_jobs 12
python experiments.py -d synthetic_random_1000 -i 8 -r 64 --gridsearch --model wwlr --n_jobs 12

python experiments.py -d synthetic_barabasi_100 -i 8 -r 64 --gridsearch --model wwlr --n_jobs 12
python experiments.py -d synthetic_barabasi_1000 -i 8 -r 64 --gridsearch --model wwlr --n_jobs 12

python experiments.py -d synthetic_watts_100 -i 8 -r 64 --gridsearch --model wwlr --n_jobs 12
python experiments.py -d synthetic_watts_1000 -i 8 -r 64 --gridsearch --model wwlr --n_jobs 12
