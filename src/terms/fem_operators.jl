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
    kpoint::FEMKpoint{T}
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
    kpoint::FEMKpoint{T}
    potential::AT
end
# We have to compute <ϕ_i|V|ψ> = ∫ ϕ_i(r) V(r) ψ(r) dr for ψ(r) = Σ_j ϕ_j(r) ψ_j, V(r) = Σ_k χ_k(r) V_k, where ϕ_i are
# the basis functions for ψ (degree m) and χ_k the basis functions for ρ (degree 2m). We expand ϕ_i(r) in the basis
# of χ_k(r) (this results in R * ψ), multiply by ϕ_j(r) (this is R .* (R * ψ)) and then integrate against V_k (this is V' * M_ρ * (R .* (R * ψ))).
# All of this is done in an efficient way (minimal number of matvec operations).
function apply!(Hψ, op::FEMRealSpaceMultiplication, ψ)
    M_ρV = get_overlap_matrix(op.basis, :ρ) * op.potential

    R = get_refinement_matrix(op.basis)
    Rψ = R * ψ

    R_copy = Complex.(R)
    R_copy.nzval .= R_copy.nzval .* Rψ[R_copy.rowval]       # SparseArrays isn't smart enough to optimize terms with structural zeros, wayyy faster than R .* Rψ

    Hψ .+= transpose(M_ρV' * R_copy)
end
function Matrix(op::FEMRealSpaceMultiplication)
    M_ρV = get_overlap_matrix(op.basis, :ρ) * op.potential
    R = get_refinement_matrix(op.basis)

    out_cols = []
    for R_col in eachcol(R)
        R_copy = Complex.(R)
        R_copy.nzval .= R_copy.nzval .* R_col[R_copy.rowval]
        push!(out_cols, transpose(M_ρV' * R_copy))
    end
    hcat(out_cols...)
end

"""
Nonlocal operator in Fourier space in Kleinman-Bylander format,
defined by its projectors P matrix and coupling terms D:
Hψ = PDP' ψ.
"""
struct NonlocalFEMOperator{T <: Real, PT, DT} <: FEMOperator
    basis::FiniteElementBasis{T}
    kpoint::FEMKpoint{T}
    # not typed, can be anything that supports PDP'ψ
    P::PT
    D::DT
end
function apply!(Hψ, op::NonlocalFEMOperator, ψ)
    mul!(Hψ, op.P, (op.D * (op.P' * ψ)), 1, 1)
end
Matrix(op::NonlocalFEMOperator) = op.P * op.D * op.P'

@doc raw"""
Laplacian operator with the usual prefactor of -1/2, i.e.
```math
Hψ = |ϕ_i><ϕ_i|-1/2 Δ|ψ>.
```
"""
struct NegHalfLaplaceFEMOperator{T <: Real} <: FEMOperator
    basis::FiniteElementBasis{T}
    kpoint::FEMKpoint{T}
end
function apply!(Hψ, op::NegHalfLaplaceFEMOperator, ψ)
    laplace_matrix = get_neg_half_laplace_matrix(op.basis, :ψ)
    if haskey(op.basis.overlap_ops, op.kpoint)
        constraint_matrix = op.basis.overlap_ops[op.kpoint].constraint_matrix
    else
        constraint_matrix = get_constraint_matrix(op.basis, op.kpoint, :ψ)
    end
    Hψ .+= constraint_matrix' * laplace_matrix * constraint_matrix * ψ
end
function Matrix(op::NegHalfLaplaceFEMOperator)
    laplace_matrix = get_neg_half_laplace_matrix(op.basis, :ψ)
    if haskey(op.basis.overlap_ops, op.kpoint)
        constraint_matrix = op.basis.overlap_ops[op.kpoint].constraint_matrix
    else
        constraint_matrix = get_constraint_matrix(op.basis, op.kpoint, :ψ)
    end
    constraint_matrix' * laplace_matrix * constraint_matrix
end

struct OverlapFEMOperator{T <: Real} <: FEMOperator
    basis::FiniteElementBasis{T}
    kpoint::FEMKpoint{T}

    constraint_matrix::AbstractMatrix
end
function apply!(Hψ, op::OverlapFEMOperator, ψ)
    overlap_matrix = get_overlap_matrix(op.basis, :ψ)
    Hψ .+= (op.constraint_matrix' * (overlap_matrix * (op.constraint_matrix * ψ)))
end
function Matrix(op::OverlapFEMOperator)
    overlap_matrix = get_overlap_matrix(op.basis, :ψ)
    op.constraint_matrix' * (overlap_matrix * op.constraint_matrix)
end