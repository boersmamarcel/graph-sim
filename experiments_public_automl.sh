python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/MUTAG.zip --online -i 4 -r 64 --categorical --model wwlr --automl &

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PROTEINS.zip --online -i 1 -r 16 --model wwlr --automl &

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/PTC_MR.zip --online -i 4 -r 64 --categorical --model wwlr --automl


python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-BINARY.zip --online -i 1 -r 64 --model wwlr --automl &
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/IMDB-MULTI.zip --online -i 1 -r 32 --model wwlr --automl

python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip --online -i 4 -r 32 --categorical --model wwlr --automl &
python experiments.py -d https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/NCI1.zip --online -i 5 -r 128 --categorical --model wwlr --automl