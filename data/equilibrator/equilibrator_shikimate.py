import numpy
import scipy.stats
from equilibrator_api import ComponentContribution, Q_, R
import json
cc = ComponentContribution()

# Shikimate pathway, added for the revision
# (fourth-pathway, non-central-carbon generalisation check).  Mirrors equilibrator_glycolysis.py:
# same conditions (pH 7.4, ionic strength 0.25 M, pMg 3.0, 298.15 K) and the
# same output schema.

default_pH = 7.4
def extract_formation_energies(metabolite_names,pH=default_pH,ionic_strength=0.25,pMg=3.0,temp=298.15):
    cc.p_h = Q_(pH)
    cc.p_mg = Q_(pMg)
    cc.ionic_strength = Q_(str(ionic_strength)+"M")
    cc.temperature = Q_(str(temp)+"K")

    compound_list = [cc.get_compound(f"bigg.metabolite:{cname}") for cname in metabolite_names]

    standard_dgf_mu, sigmas_fin, sigmas_inf = zip(*map(cc.standard_dg_formation, compound_list))
    standard_dgf_mu = numpy.array(standard_dgf_mu)
    sigmas_fin = numpy.array(sigmas_fin)
    sigmas_inf = numpy.array(sigmas_inf)

    delta_dgf_list = numpy.array([
        cpd.transform(cc.p_h, cc.ionic_strength, cc.temperature, cc.p_mg).m_as("kJ/mol")
        for cpd in compound_list
    ])
    standard_dgf_prime_mu = standard_dgf_mu + delta_dgf_list

    standard_dgf_cov = sigmas_fin @ sigmas_fin.T + 1e6 * sigmas_inf @ sigmas_inf.T

    n_H = numpy.array([c.atom_bag['H'] if 'H' in c.atom_bag.keys() else 0 for c in compound_list])
    standard_dgf_prime_mu_adjusted = standard_dgf_prime_mu - R.magnitude*temp*pH*n_H

    return standard_dgf_prime_mu,standard_dgf_prime_mu_adjusted,standard_dgf_cov

metabolite_names = [
    "e4p",
    "pep",
    "h2o",
    "2dda7p",
    "pi",
    "3dhq",
    "3dhsk",
    "nadph",
    "nadp",
    "skm",
    "atp",
    "adp",
    "skm5p",
    "3psme",
    "chor"
]

standard_dgf_prime_mu,standard_dgf_prime_mu_adjusted,standard_dgf_cov = extract_formation_energies(metabolite_names)

output = {
    "metabolite_names": metabolite_names,
    "dgf_prime_mu": standard_dgf_prime_mu.tolist(),
    "dgf_prime_mu_adj": standard_dgf_prime_mu_adjusted.tolist(),
    "dgf_prime_cov": standard_dgf_cov.tolist()
}
with open('equilibrator_shikimate.json', 'w') as f:
    json.dump(output, f)
print("wrote equilibrator_shikimate.json")
