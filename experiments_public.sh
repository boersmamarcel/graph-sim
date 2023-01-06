python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip --online -i 8 -r 128 --categorical --model wwlr --n_jobs 12 --gridsearch

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip --online -i 8 -r 128 --gridsearch --model wwlr --n_jobs 12
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip --online -i 8 -r 128 --gridsearch --categorical --model wwlr --n_jobs 12

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/ENZYMES.zip --online -i 8 -r 128 --gridsearch --model wwlr --n_jobs 12
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/ENZYMES.zip --online -i 8 -r 128 --gridsearch --categorical --model wwlr --n_jobs 12

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PTC_MR.zip --online -i 8 -r 128 --gridsearch --categorical --model wwlr --n_jobs 12

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip --online -i 8 -r 128 --categorical --model wwlr --n_jobs 12 --gridsearch

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-MULTI.zip --online -i 8 -r 128 --gridsearch --model wwlr --n_jobs 12
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-BINARY.zip --online -i 8 -r 128 --gridsearch --model wwlr --n_jobs 12

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 8 -r 128 --categorical --model wwlr --n_jobs 12 --gridsearch

#python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip --online -i 1 -r 64 --categorical --automl &
#python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 5 -r 256 --categorical --output r256 &
#python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 5 -r 512 --categorical --output r512
#python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 5 -r 256 --categorical --representations r256 --random_refs --output r256_random &
#python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 5 -r 512 --categorical --representations r512 --random_refs --output r512_random &
#python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 5 -r 256 --categorical --representations r256 --automl --output r256_automl &
#python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 5 -r 512 --categorical --representations r512 --automl --output r512_automl

