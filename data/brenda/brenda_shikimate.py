import pandas as pd
from brendapyrser import BRENDA

# Pentose phosphate pathway, added for the revision
# (fourth-pathway (non-central-carbon) generalisation check). Mirrors brenda_glycolysis.py.
#
# The BRENDA 2023.1 flat file is licensed and therefore not stored in this
# repository; point dataFile at a local copy.
dataFile = "/Users/nguye223/Library/CloudStorage/OneDrive-QueenslandUniversityofTechnology/Manuscript/Macsys/CCM/250411_Michael_BI_Metabolism/brenda/brenda_2023_1.txt"

enzymes = [
    "2.5.1.54",  # DAHP synthase (ARO3/ARO4)
    "4.2.3.4",   # 3-dehydroquinate synthase (ARO1)
    "4.2.1.10",  # 3-dehydroquinate dehydratase (ARO1)
    "1.1.1.25",  # shikimate dehydrogenase (ARO1)
    "2.7.1.71",  # shikimate kinase (ARO1)
    "2.5.1.19",  # EPSP synthase (ARO1)
    "4.2.3.5",   # chorismate synthase (ARO2)
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
df.to_csv('KM_values_shikimate.csv', index=False)

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
df.to_csv('kcat_values_shikimate.csv', index=False)

print("kcat records:", len(df))
print("done")
