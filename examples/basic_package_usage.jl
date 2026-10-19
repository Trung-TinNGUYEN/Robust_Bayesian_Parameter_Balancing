using RobustBayesianParameterBalancing
using LinearAlgebra

prior = load_parameter_balancing_prior()

Q = Matrix{Float64}(I, 2, 2)
x = [0.1, -0.2]
Cx = 0.05 .* Matrix{Float64}(I, 2, 2)
q_prior = zeros(2)
C_prior = Matrix{Float64}(I, 2, 2)

result = balance_parameters((
    Q = Q,
    x = x,
    Cx = Cx,
    q_prior = q_prior,
    C_prior = C_prior,
    parameter_names = ["mu0_A", "mu0_B"],
    observable_names = ["obs_A", "obs_B"],
    quantity_types = ["standard chemical potential", "standard chemical potential"],
    units = ["kJ/mol", "kJ/mol"]
), prior)

summary = export_balanced_table(result; format = :dataframe)
println(summary)
