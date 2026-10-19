import pandas as pd
from brendapyrser import BRENDA

# Pentose phosphate pathway, added for the revision
# (third-pathway generalisation check). Mirrors brenda_glycolysis.py.
#
# The BRENDA 2023.1 flat file is licensed and therefore not stored in this
# repository; point dataFile at a local copy.
dataFile = "/Users/nguye223/Library/CloudStorage/OneDrive-QueenslandUniversityofTechnology/Manuscript/Macsys/CCM/250411_Michael_BI_Metabolism/brenda/brenda_2023_1.txt"

enzymes = [
    "1.1.1.49",  # glucose-6-phosphate dehydrogenase (ZWF1)
    "3.1.1.31",  # 6-phosphogluconolactonase (SOL3/SOL4)
    "1.1.1.44",  # 6-phosphogluconate dehydrogenase (GND1/GND2)
    "5.3.1.6",   # ribose-5-phosphate isomerase (RKI1)
    "5.1.3.1",   # ribulose-phosphate 3-epimerase (RPE1)
    "2.2.1.1",   # transketolase (TKL1/TKL2)
    "2.2.1.2",   # transaldolase (TAL1)
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
    for k, v in r.KMvalues.items():
        for d in v:
            for s in d["species"]:
                ec_codes.append(r.ec_number)
                enzyme_names.append(r.name)
                KMs.append(d["value"])
                substrates.append(k)
                species.append(s)
                metadata.append(d["meta"])

df = pd.DataFrame({"ec_code": ec_codes, "name": enzyme_names, "KM": KMs,
                   "substrate": substrates, "species": species, "metadata": metadata})
df.to_csv('KM_values_pentose_phosphate.csv', index=False)

# Extract kcat values
ec_codes = []
enzyme_names = []
kcats = []
substrates = []
species = []
metadata = []
for r in reactions:
    for k, v in r.Kcatvalues.items():
        for d in v:
            for s in d["species"]:
                ec_codes.append(r.ec_number)
                enzyme_names.append(r.name)
                kcats.append(d["value"])
                substrates.append(k)
                species.append(s)
                metadata.append(d["meta"])

df = pd.DataFrame({"ec_code": ec_codes, "name": enzyme_names, "kcat": kcats,
                   "substrate": substrates, "species": species, "metadata": metadata})
df.to_csv('kcat_values_pentose_phosphate.csv', index=False)

print("kcat records:", len(df))
print("done")
