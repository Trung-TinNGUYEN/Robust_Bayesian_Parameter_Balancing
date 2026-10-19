import numpy as np
from matplotlib import pyplot as plt
from brendapyrser import BRENDA
import pandas as pd

dataFile = "brenda_2023_1.txt"
enzymes = [
    "2.3.3.1",
    "4.2.1.3",
    "1.1.1.41",
    "1.2.4.2",
    "2.3.1.61",
    "1.8.1.4",
    "6.2.1.5",
    "1.3.5.1",
    "4.2.1.2",
    "1.1.1.37"
]

brenda = BRENDA(dataFile)
reactions = [brenda.reactions.get_by_id(e) for e in enzymes]

# Extract KM values
ec_codes = []
enzyme_names = []
KMs = []
substrates = []
species = []
metadata = []
for r in reactions:
    for k,v in r.KMvalues.items():
        for d in v:
            for s in d["species"]:
                ec_codes.append(r.ec_number)
                enzyme_names.append(r.name)
                KMs.append(d["value"])
                substrates.append(k)
                species.append(s)
                metadata.append(d["meta"])

df = pd.DataFrame({"ec_code": ec_codes, "name": enzyme_names, "KM": KMs, "substrate": substrates, "species": species, "metadata": metadata})
df.to_csv('KM_values_tca_cycle.csv', index=False)

# Extract kcat values
ec_codes = []
enzyme_names = []
kcats = []
substrates = []
species = []
metadata = []
for r in reactions:
    for k,v in r.Kcatvalues.items():
        for d in v:
            for s in d["species"]:
                ec_codes.append(r.ec_number)
                enzyme_names.append(r.name)
                kcats.append(d["value"])
                substrates.append(k)
                species.append(s)
                metadata.append(d["meta"])

df = pd.DataFrame({"ec_code": ec_codes, "name": enzyme_names, "kcat": kcats, "substrate": substrates, "species": species, "metadata": metadata})
df.to_csv('kcat_values_tca_cycle.csv', index=False)