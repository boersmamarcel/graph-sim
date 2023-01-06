from datagenerator.generator import DataGenerator

def create_network(n):
    dg = DataGenerator("data/synthetic_" + str(n) + "/" , n)
    dg.smallClasses(50)

    random = DataGenerator("data/synthetic_random_" + str(n) + "/", n)
    random.sameClassDifferentSettingsRandom(50)

    barabasi = DataGenerator("data/synthetic_barabasi_" + str(n) + "/", n)
    barabasi.sameClassDifferentSettingsBarabasi(50)
    
    watts = DataGenerator("data/synthetic_watts_" + str(n) + "/", n)
    watts.sameClassDifferentSettingsWattsStrogatz(50)


try:
    from joblib import Parallel, delayed
    
    Parallel(n_jobs=4, backend='threading')(delayed(create_network)(n) for n in [100, 1000])
except Exception as e:
    print("FAILURE")

