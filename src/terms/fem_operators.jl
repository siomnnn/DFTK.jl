### Linear operators operating on FEM-discretized quantities in real space

using Ferrite

"""
Linear operators that act on FEM-discretized real-space quantities.
The main entry point is `apply!(out, op, in)` which performs the operation `out += op*in`
where `out` and `in` are vectors whose indices correspond to the ordering defined by the
`dof_handler` of the basis.

They also implement `mul!` and `Matrix(op)` for exploratory use.

In order for pretty much any `FEMOperator` to work correctly, all values in ψ corresponding
to periodic `prescribed_dofs` in the constraint handler must be set to zero. This is done by
using the `apply_bc!` and `remove_bc!` functions depending on whether we want to sum the values
at every periodic dof (`apply_bc!`) or just overwrite the non-"main" dofs with zeros (`remove_bc!`).
Manually playing around with this is possible, but not recommended (usually, boundary conditions are
dealt with by higher-level functions).
"""
abstract type FEMOperator end
# FEMOperator currently only contain a field `basis`, as `kpoint`s are not yet implemented for FEM.

function LinearAlgebra.mul!(Hψ::AbstractVector, op::FEMOperator, ψ::AbstractVector)
    # Only real-space transformations are possible with finite elements,
    # so there is no Fourier component.
    Hψ .= 0
    apply!(Hψ, op, ψ)
    Hψ
end
function LinearAlgebra.mul!(Hψ::AbstractMatrix, op::FEMOperator, ψ::AbstractMatrix)
    @views for i = 1:size(ψ, 2)
        mul!(Hψ[:, i], op, ψ[:, i])
    end
    Hψ
end
Base.:*(op::FEMOperator, ψ) = mul!(similar(ψ), op, ψ)

"""
Noop operation: don't do anything.
Useful for energy terms that don't depend on the orbitals at all (eg nuclei-nuclei interaction).
"""
struct NoopFEMOperator{T <: Real} <: FEMOperator
    basis::FiniteElementBasis{T}
end
apply!(Hψ, op::NoopFEMOperator, ψ) = nothing
function Matrix(op::NoopFEMOperator)
    n_dofs = get_n_free_dofs(op.basis, :ψ)
    zeros(eltype(op.basis), n_dofs, n_dofs)
end

# Contrary to the Fourier space version, we need to take into account that the
# FEM basis functions are not orthonormal, so we cannot just multiply by the potential.
# At the same time, constructing the full matrix is unnecessarily expensive and not
# needed for diagonalization or energy computation. Instead, we only compute the
# "matrix elements" <ϕ_i|V|ψ> using the Ferrite features designed for FEM load vector assembly.
"""
Real space multiplication by a potential:
```math
Hψ = |ϕ_i><ϕ_i|V|ψ>.
```
"""
struct FEMRealSpaceMultiplication{T <: Real, AT <: AbstractArray} <: FEMOperator
    basis::FiniteElementBasis{T}
    potential::AT
end
function apply!(Hψ, op::FEMRealSpaceMultiplication, ψ)
    dof_handler = get_dof_handler(op.basis, :ψ)
    cell_values = get_cell_values(op.basis, :ψ)
    constraint_handler = get_constraint_handler(op.basis, :ψ)

    out = zeros(eltype(op.basis), ndofs(dof_handler))

    n_basefuncs = getnbasefunctions(cell_values)
    fe = zeros(n_basefuncs)

    # all of these remain constant when reinit-ing cell_values in the case of a Lagrange basis
    n_quad = getnquadpoints(cell_values)
    ϕ_evals = shape_value.([cell_values], 1:n_quad, (1:n_basefuncs)')

    for cell in CellIterator(dof_handler)
        reinit!(cell_values, cell)
        fill!(fe, 0)

        periodic_cell_dofs = apply_inverse_constraint_map(op.basis, celldofs(cell), :ψ)

        pot_interpol = ϕ_evals * op.potential[periodic_cell_dofs]
        ψ_interpol = ϕ_evals * ψ[periodic_cell_dofs]
        dΩ = getdetJdV.([cell_values], 1:n_quad)
        
        for i in 1:n_basefuncs
            fe[i] += (ϕ_evals[:, i] .* ψ_interpol .* pot_interpol)' * dΩ
        end
    
        assemble!(out, celldofs(cell), fe)
    end

    apply_bc!(out, constraint_handler)

    Hψ .+= out[get_free_dofs(op.basis, :ψ)]
end
function Matrix(op::FEMRealSpaceMultiplication)
    dof_handler = get_dof_handler(op.basis, :ψ)
    constr_handler = get_constraint_handler(op.basis, :ψ)
    cell_values = get_cell_values(op.basis, :ψ)

    H = allocate_matrix(dof_handler, constr_handler)

    n_basefuncs = getnbasefunctions(cell_values)
    Ke = zeros(complex(eltype(op.basis)), n_basefuncs, n_basefuncs)
    
    assembler = start_assemble(H)

    # all of these remain constant when reinit-ing cell_values in the case of a Lagrange basis
    n_quad = getnquadpoints(cell_values)
    ϕ_evals = shape_value.([cell_values], 1:n_quad, (1:n_basefuncs)')

    # TODO: is parallelization possible even though we are reinit-ing cell_values?
    for cell in CellIterator(dof_handler)
        reinit!(cell_values, cell)
        fill!(Ke, 0)

        periodic_cell_dofs = apply_inverse_constraint_map(op.basis, celldofs(cell), :ψ)

        pot_interpol = ϕ_evals * op.potential[periodic_cell_dofs]
        dΩ = getdetJdV.([cell_values], 1:n_quad)

        for i in 1:n_basefuncs, j in 1:n_basefuncs
            Ke[i, j] += (ϕ_evals[:, i] .* ϕ_evals[:, j] .* pot_interpol)' * dΩ
        end
    
        assemble!(assembler, celldofs(cell), Ke)
    end

    Ferrite.apply!(H, constr_handler)

    free_dofs = get_free_dofs(op.basis, :ψ)
    H[free_dofs, free_dofs]
end

#"""
#Nonlocal operator in Fourier space in Kleinman-Bylander format,
#defined by its projectors P matrix and coupling terms D:
#Hψ = PDP' ψ.
#"""
#struct NonlocalOperator{T <: Real, PT, DT} <: RealFourierOperator
#    basis::PlaneWaveBasis{T}
#    kpoint::Kpoint{T}
#    # not typed, can be anything that supports PDP'ψ
#    P::PT
#    D::DT
#end
#function apply!(Hψ, op::NonlocalOperator, ψ)
#    mul!(Hψ.fourier, op.P, (op.D * (op.P' * ψ.fourier)), 1, 1)
#end
#Matrix(op::NonlocalOperator) = op.P * op.D * op.P'
#
@doc raw"""
Laplacian operator with the usual prefactor of -1/2, i.e.
```math
Hψ = |ϕ_i><ϕ_i|-1/2 Δ|ψ>.
```
"""
struct NegHalfLaplaceFEMOperator{T <: Real} <: FEMOperator
    basis::FiniteElementBasis{T}
end
function apply!(Hψ, op::NegHalfLaplaceFEMOperator, ψ)
    laplace_matrix = get_neg_half_laplace_matrix(op.basis, :ψ)
    if !isnothing(laplace_matrix)
        Hψ .+= laplace_matrix * ψ
        return
    end

    dof_handler = get_dof_handler(op.basis, :ψ)
    cell_values = get_cell_values(op.basis, :ψ)
    out = zeros(eltype(op.basis), ndofs(dof_handler))

    n_basefuncs = getnbasefunctions(cell_values)
    fe = zeros(n_basefuncs)

    n_quad = getnquadpoints(cell_values)

    for cell in CellIterator(dof_handler)
        reinit!(cell_values, cell)
        fill!(fe, 0)

        periodic_cell_dofs = apply_inverse_constraint_map(op.basis, celldofs(cell), :ψ)

        ∇ϕ_evals = shape_gradient.([cell_values], 1:n_quad, (1:n_basefuncs)')
        ψ_interpol = ∇ϕ_evals * ψ[periodic_cell_dofs]
        dΩ = getdetJdV.([cell_values], 1:n_quad)

        for i in 1:n_basefuncs
            fe[i] += 0.5 * (∇ϕ_evals[:, i] .⋅ ψ_interpol) ⋅ dΩ
        end
    
        assemble!(out, celldofs(cell), fe)
    end

    apply_bc!(out, get_constraint_handler(op.basis, :ψ))

    Hψ .+= out[get_free_dofs(op.basis, :ψ)]
    return
end
function Matrix(op::NegHalfLaplaceFEMOperator)
    if !isnothing(op.basis.neg_half_laplacian)
        return op.basis.neg_half_laplacian
    end
    init_neg_half_laplace_matrix(op.basis.discretization, :ψ)
end