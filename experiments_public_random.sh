python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip --online -i 8 -r 128 --categorical --model wwlr --n_jobs 12 --gridsearch --random_refs

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip --online -i 8 -r 128 --gridsearch --model wwlr --n_jobs 12 --random_refs

# if time permits:
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip --online -i 8 -r 128 --gridsearch --categorical --model wwlr --n_jobs 12 --random_refs

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/ENZYMES.zip --online -i 8 -r 128 --gridsearch --model wwlr --n_jobs 12 --random_refs
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/ENZYMES.zip --online -i 8 -r 128 --gridsearch --categorical --model wwlr --n_jobs 12 --random_refs

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PTC_MR.zip --online -i 8 -r 128 --gridsearch --categorical --model wwlr --n_jobs 12 --random_refs

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip --online -i 8 -r 128 --categorical --model wwlr --n_jobs 12 --gridsearch --random_refs

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-MULTI.zip --online -i 8 -r 128 --gridsearch --model wwlr --n_jobs 12 --random_refs
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-BINARY.zip --online -i 8 -r 128 --gridsearch --model wwlr --n_jobs 12 --random_refs

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 8 -r 128 --categorical --model wwlr --n_jobs 12 --gridsearch --random_refs
