### Linear operators operating on FEM-discretized quantities in real space

using Ferrite

"""
Linear operators that act on FEM-discretized real-space quantities.
The main entry point is `apply!(out, op, in)` which performs the operation `out += op*in`
where `out` and `in` are vectors whose indices correspond to the ordering defined by the
`dof_handler` of the basis.
They also implement `mul!` and `Matrix(op)` for exploratory use.
"""
abstract type FEMOperator end
# FEMOperator currently only contain a field `basis`, as `kpoint`s are not yet implemented for FEM.

#function LinearAlgebra.mul!(Hψ::AbstractVector, op::FEMOperator, ψ::AbstractVector)
#    ψ_real = ifft(op.basis, op.kpoint, ψ)
#    Hψ_fourier = similar(ψ)
#    Hψ_real = similar(ψ_real)
#    Hψ_fourier .= 0
#    Hψ_real .= 0
#    apply!((; real=Hψ_real, fourier=Hψ_fourier),
#           op,
#           (; real=ψ_real, fourier=ψ))
#    Hψ .= Hψ_fourier .+ fft(op.basis, op.kpoint, Hψ_real)
#    Hψ
#end
#function LinearAlgebra.mul!(Hψ::AbstractMatrix, op::FEMOperator, ψ::AbstractMatrix)
#    @views for i = 1:size(ψ, 2)
#        mul!(Hψ[:, i], op, ψ[:, i])
#    end
#    Hψ
#end
#Base.:*(op::RealFourierOperator, ψ) = mul!(similar(ψ), op, ψ)

"""
Noop operation: don't do anything.
Useful for energy terms that don't depend on the orbitals at all (eg nuclei-nuclei interaction).
"""
struct NoopFEMOperator{T <: Real} <: FEMOperator
    basis::FiniteElementBasis{T}
end
apply!(Hψ, op::NoopFEMOperator, ψ) = nothing
function Matrix(op::NoopFEMOperator)
    n_dofs = getndofs(op.basis)
    zeros(eltype(op.basis), n_dofs, n_dofs)
end

"""
Real space multiplication by a potential:
```math
(Hψ)(r) = V(r) ψ(r).
```
"""
struct FEMRealSpaceMultiplication{T <: Real, AT <: AbstractArray} <: FEMOperator
    basis::FiniteElementBasis{T}
    potential::AT
end
function apply!(Hψ, op::FEMRealSpaceMultiplication, ψ)
    Hψ .+= op.potential .* ψ
end
function Matrix(op::FEMRealSpaceMultiplication)
    # V(ϕ1, ϕ2) = <ϕ1|V|ϕ2> = ∫ conj(ϕ1(r)) V(r) ϕ2(r) dr
    dof_handler = getdofhandler(op.basis)
    constr_handler = getconstrainthandler(op.basis)
    cell_values = getcellvalues(op.basis)

    H = allocate_matrix(dof_handler, constr_handler)

    n_basefuncs = getnbasefunctions(cell_values)
    Ke = zeros(complex(eltype(op.basis)), n_basefuncs, n_basefuncs)
    
    assembler = start_assemble(H)

    # TODO: is parallelization possible even though we are reinit-ing cell_values?
    for cell in CellIterator(dof_handler)
        reinit!(cell_values, cell)

        assemble_element!(Ke, cell_values, op.potential[celldofs(cell)])
    
        assemble!(assembler, celldofs(cell), Ke)
    end
    H
end

function assemble_element!(Ke, cv, potential)
    n_basefuncs = getnbasefunctions(cv)
    fill!(Ke, 0)

    n_quad = getnquadpoints(cv)
    ϕ_evals = shape_value.([cv], 1:n_quad, (1:n_basefuncs)')
    pot_interpol = ϕ_evals * potential

    # TODO: maybe vectorize
    for q_point in 1:n_quad
        dΩ = getdetJdV(cv, q_point)

        for i in 1:n_basefuncs, j in 1:n_basefuncs
            Ke[i, j] += pot_interpol[q_point] * ϕ_evals[q_point, i] * ϕ_evals[q_point, j] * dΩ
        end
    end
    return Ke
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
#@doc raw"""
#Nonlocal "divAgrad" operator ``-½ ∇ ⋅ (A ∇)`` where ``A`` is a scalar field on the
#real-space grid. The ``-½`` is included, such that this operator is a generalisation of the
#kinetic energy operator (which is obtained for ``A=1``).
#"""
#struct DivAgradOperator{T <: Real, AT} <: RealFourierOperator
#    basis::PlaneWaveBasis{T}
#    kpoint::Kpoint{T}
#    A::AT
#end
#function apply!(Hψ, op::DivAgradOperator, ψ;
#                ψ_scratch=zeros(complex(eltype(op.basis)), op.basis.fft_size...))
#    # TODO: Performance improvements: Unscaled plans, avoid remaining allocations
#    #       (which are only on the small k-point-specific Fourier grid
#    G_plus_k = [[p[α] for p in Gplusk_vectors_cart(op.basis, op.kpoint)] for α = 1:3]
#    for α = 1:3
#        ∂αψ_real = ifft!(ψ_scratch, op.basis, op.kpoint, im .* G_plus_k[α] .* ψ.fourier)
#        A∇ψ      = fft(op.basis, op.kpoint, ∂αψ_real .* op.A)
#        Hψ.fourier .-= im .* G_plus_k[α] .* A∇ψ ./ 2
#    end
#end
## TODO Implement  Matrix(op::DivAgrad)
