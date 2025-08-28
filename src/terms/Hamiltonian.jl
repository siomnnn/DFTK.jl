### A Hamiltonian is composed of blocks (kpoints), which have a list of RealFourierOperator
# corresponding to each term
# This is the "high-level" interface, provided for convenience

abstract type HamiltonianBlock{B <: AbstractBasis} end

# Generic HamiltonianBlock
struct GenericHamiltonianBlock{B <: PlaneWaveBasis} <: HamiltonianBlock{B}
    basis::B
    kpoint::Kpoint

    # The operators are vectors of RealFourierOperator,
    # not typed because of type invariance issues.
    operators::Vector  # the original list of RealFourierOperator
                       # (as many as there are terms), kept for easier exploration
    optimized_operators::Vector  # Optimized list of RealFourierOperator, for application

    scratch  # dummy field
end

# FEM only does generic blocks for now  
struct GenericHamiltonianBlockFEM{B <: FiniteElementBasis} <: HamiltonianBlock{B}
    basis::B
    kpoint::FEMKpoint

    # The operators are vectors of FEMOperator,
    # not typed because of type invariance issues.
    operators::Vector
    optimized_operators::Vector  # Optimized list of FEMOperator, for application

    scratch  # dummy field
end

# More optimized HamiltonianBlock for the important case of a DFT Hamiltonian
struct DftHamiltonianBlock{B <: PlaneWaveBasis} <: HamiltonianBlock{B}
    basis::B
    kpoint::Kpoint
    operators::Vector

    # Individual operators for easy access
    fourier_op::FourierMultiplication
    local_op::RealSpaceMultiplication
    nonlocal_op::Union{Nothing,NonlocalOperator}
    divAgrad_op::Union{Nothing,DivAgradOperator}

    scratch  # Pre-allocated scratch arrays for fast application
end

function HamiltonianBlock(basis::PlaneWaveBasis, kpoint, operators; scratch=nothing)
    optimized_operators = optimize_operators(operators)
    fourier_ops  = filter(o -> o isa FourierMultiplication,   optimized_operators)
    real_ops     = filter(o -> o isa RealSpaceMultiplication, optimized_operators)
    nonlocal_ops = filter(o -> o isa NonlocalOperator,        optimized_operators)
    divAgrad_ops = filter(o -> o isa DivAgradOperator,        optimized_operators)

    n_ops_grouped = length(fourier_ops) + length(real_ops) + length(nonlocal_ops) + length(divAgrad_ops)
    is_dft_ham = (   length(fourier_ops) == 1 && length(real_ops) == 1
                  && length(nonlocal_ops) < 2 && length(divAgrad_ops) < 2
                  && n_ops_grouped == length(optimized_operators))
    if is_dft_ham
        scratch = @something scratch _ham_allocate_scratch(basis)
        nonlocal_op = isempty(nonlocal_ops) ? nothing : only(nonlocal_ops)
        divAgrad_op = isempty(divAgrad_ops) ? nothing : only(divAgrad_ops)
        DftHamiltonianBlock(basis, kpoint, operators,
                            only(fourier_ops), only(real_ops),
                            nonlocal_op, divAgrad_op, scratch)
    else
        GenericHamiltonianBlock(basis, kpoint, operators, optimized_operators, nothing)
    end
end
function _ham_allocate_scratch(basis::PlaneWaveBasis{T}) where {T}
    [(; ψ_reals=zeros_like(G_vectors(basis), complex(T), basis.fft_size...))
     for _ = 1:Threads.nthreads()]
end
function _ham_allocate_scratch(basis::FiniteElementBasis{T}) where {T}
    [(; ψ_scratch=zeros(complex(T), get_n_free_dofs(basis, :ψ)))
     for _ = 1:Threads.nthreads()]
end

function HamiltonianBlock(basis::FiniteElementBasis, kpoint::FEMKpoint, operators; scratch=nothing)
    optimized_operators = optimize_operators(operators)
    GenericHamiltonianBlockFEM(basis, kpoint, operators, optimized_operators, nothing)
end

Base.:*(H::HamiltonianBlock, ψ) = mul!(similar(ψ), H, ψ)
Base.eltype(block::HamiltonianBlock) = complex(eltype(block.basis))
Base.size(block::HamiltonianBlock, i::Integer) = i < 3 ? size(block)[i] : 1
function Base.size(block::HamiltonianBlock{B}) where {B <: PlaneWaveBasis}
    n_G = length(G_vectors(block.basis, block.kpoint))
    (n_G, n_G)
end
function Base.size(block::HamiltonianBlock{B}) where {B <: FiniteElementBasis}
    n_dofs = get_n_free_dofs(block.basis, :ψ)
    (n_dofs, n_dofs)
end
function random_orbitals(hamk::HamiltonianBlock, howmany::Integer)
    random_orbitals(hamk.basis, hamk.kpoint, howmany)
end

import Base: Matrix, Array
Array(block::HamiltonianBlock)  = Matrix(block)
Matrix(block::HamiltonianBlock) = sum(Matrix, block.operators)
Matrix(block::GenericHamiltonianBlock) = sum(Matrix, block.optimized_operators)
Matrix(block::GenericHamiltonianBlockFEM) = sum(Matrix, block.optimized_operators)

struct Hamiltonian{B <: AbstractBasis}
    basis::B
    blocks::Vector{<:HamiltonianBlock{B}}
end

Base.getindex(ham::Hamiltonian, index) = ham.blocks[index]

function LinearAlgebra.mul!(Hψ, H::Hamiltonian, ψ)
    for ik = 1:length(H.basis.kpoints)
        mul!(Hψ[ik], H.blocks[ik], ψ[ik])
    end
    Hψ
end
function Base.:*(H::Hamiltonian, ψ)
    # This allocates new memory for the result of promoted eltype
    result = one(eltype(H.basis)) * ψ
    mul!(result, H, ψ)
end

# Loop through bands, IFFT to get ψ in real space, loop through terms, FFT and accumulate into Hψ
# For the common DftHamiltonianBlock there is an optimized version below
@views @timing "Hamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                        H::GenericHamiltonianBlock,
                                                                        ψ::AbstractArray)
    function allocate_local_storage()
        T = eltype(H.basis)
        (; Hψ_fourier = similar(Hψ[:, 1]),
           ψ_real  = similar(ψ, complex(T), H.basis.fft_size...),
           Hψ_real = similar(Hψ, complex(T), H.basis.fft_size...))
    end
    parallel_loop_over_range(1:size(ψ, 2); allocate_local_storage) do iband, storage
        to = TimerOutput()  # Thread-local timer output

        # Take ψi, IFFT it to ψ_real, apply each term to Hψ_fourier and Hψ_real, and add it
        # to Hψ.
        storage.Hψ_real .= 0
        storage.Hψ_fourier .= 0
        ifft!(storage.ψ_real, H.basis, H.kpoint, ψ[:, iband])
        for op in H.optimized_operators
            @timeit to "$(nameof(typeof(op)))" begin
                apply!((; fourier=storage.Hψ_fourier, real=storage.Hψ_real),
                       op,
                       (; fourier=ψ[:, iband], real=storage.ψ_real))
            end
        end
        Hψ[:, iband] .= storage.Hψ_fourier
        fft!(storage.Hψ_fourier, H.basis, H.kpoint, storage.Hψ_real)
        Hψ[:, iband] .+= storage.Hψ_fourier

        if Threads.threadid() == 1
            merge!(DFTK.timer, to; tree_point=[t.name for t in DFTK.timer.timer_stack])
        end
    end

    Hψ
end

@views @timing "Hamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                        H::GenericHamiltonianBlockFEM,
                                                                        ψ::AbstractArray)
    function allocate_local_storage()
        T = eltype(H.basis)
        (; Hψ_band = similar(Hψ, complex(T), get_n_free_dofs(H.basis, :ψ)))
    end
    parallel_loop_over_range(1:size(ψ, 2); allocate_local_storage) do iband, storage
        to = TimerOutput()  # Thread-local timer output

        storage.Hψ_band .= 0
        for op in H.optimized_operators
            @timeit to "$(nameof(typeof(op)))" begin
                apply!(storage.Hψ_band, op, ψ[:, iband])
            end
        end
        Hψ[:, iband] .= storage.Hψ_band

        if Threads.threadid() == 1
            merge!(DFTK.timer, to; tree_point=[t.name for t in DFTK.timer.timer_stack])
        end
    end

    Hψ
end

# Fast version, specialized on DFT models. Minimizes the number of FFTs and allocations
@views @timing "DftHamiltonian multiplication" function LinearAlgebra.mul!(Hψ::AbstractArray,
                                                                           H::DftHamiltonianBlock,
                                                                           ψ::AbstractArray)
    n_bands = size(ψ, 2)
    iszero(n_bands) && return Hψ  # Nothing to do if ψ empty
    have_divAgrad = !isnothing(H.divAgrad_op)

    # Notice that we use unnormalized plans for extra speed
    potential = H.local_op.potential / prod(H.basis.fft_size)

    parallel_loop_over_range(1:n_bands, H.scratch) do iband, storage
        to = TimerOutput()  # Thread-local timer output
        ψ_real = storage.ψ_reals

        @timeit to "local+kinetic" begin
            ifft!(ψ_real, H.basis, H.kpoint, ψ[:, iband]; normalize=false)
            ψ_real .*= potential
            fft!(Hψ[:, iband], H.basis, H.kpoint, ψ_real; normalize=false)  # overwrites ψ_real
            Hψ[:, iband] .+= H.fourier_op.multiplier .* ψ[:, iband]
        end

        if have_divAgrad
            @timeit to "divAgrad" begin
                apply!((; fourier=Hψ[:, iband], real=nothing),
                       H.divAgrad_op,
                       (; fourier=ψ[:, iband], real=nothing);
                       ψ_scratch=ψ_real)
            end
        end

        if Threads.threadid() == 1
            merge!(DFTK.timer, to; tree_point=[t.name for t in DFTK.timer.timer_stack])
        end

        synchronize_device(H.basis.architecture)
    end

    # Apply the nonlocal operator.
    if !isnothing(H.nonlocal_op)
        @timing "nonlocal" begin
            apply!((; fourier=Hψ, real=nothing),
                   H.nonlocal_op,
                   (; fourier=ψ, real=nothing))
        end
    end

    Hψ
end


"""
Get energies and Hamiltonian
kwargs is additional info that might be useful for the energy terms to precompute
(eg the density ρ)
"""
@timing function energy_hamiltonian(basis::PlaneWaveBasis, ψ, occupation; kwargs...)
    # it: index into terms, ik: index into kpoints
    @timing "ene_ops" ene_ops_arr = [ene_ops(term, basis, ψ, occupation; kwargs...)
                                     for term in basis.terms]
    term_names = [string(nameof(typeof(term))) for term in basis.model.term_types]
    energy_values  = [eh.E for eh in ene_ops_arr]
    operators = [eh.ops for eh in ene_ops_arr]         # operators[it][ik]

    # flatten the inner arrays in case a term returns more than one operator
    function flatten(arr)
        ret = []
        for a in arr
            if a isa RealFourierOperator
                push!(ret, a)
            else
                push!(ret, a...)
            end
        end
        ret
    end
    scratch = _ham_allocate_scratch(basis)
    hks_per_k = [flatten([blocks[ik] for blocks in operators])
                 for ik = 1:length(basis.kpoints)]      # hks_per_k[ik][it]
    ham = Hamiltonian(basis, [HamiltonianBlock(basis, kpt, hks; scratch)
                              for (hks, kpt) in zip(hks_per_k, basis.kpoints)])
    energies = Energies(term_names, energy_values)
    (; energies, ham)
end

@timing function energy_hamiltonian(basis::FiniteElementBasis, ψ, occupation; kwargs...)
    # it: index into terms, ik: index into kpoints
    @timing "ene_ops" ene_ops_arr = [ene_ops(term, basis, ψ, occupation; kwargs...)
                                     for term in basis.terms]
    term_names = [string(nameof(typeof(term))) for term in basis.model.term_types]
    energy_values  = [eh.E for eh in ene_ops_arr]
    operators = [eh.ops for eh in ene_ops_arr]         # operators[it][ik]

    # flatten the inner arrays in case a term returns more than one operator
    function flatten(arr)
        ret = []
        for a in arr
            if a isa FEMOperator
                push!(ret, a)
            else
                push!(ret, a...)
            end
        end
        ret
    end
    scratch = _ham_allocate_scratch(basis)              # doesn't do anything yet since no DFT-specific blocks
    hks_per_k = [flatten([blocks[ik] for blocks in operators])
                 for ik = 1:length(basis.kpoints)]      # hks_per_k[ik][it]
    ham = Hamiltonian(basis, [HamiltonianBlock(basis, kpt, hks; scratch)
                              for (hks, kpt) in zip(hks_per_k, basis.kpoints)])
    energies = Energies(term_names, energy_values)
    (; energies, ham)
end

"""
Faster version than energy_hamiltonian for cases where only the energy is needed.
"""
@timing function energy(basis::AbstractBasis, ψ, occupation; kwargs...)
    energy_values = [energy(term, basis, ψ, occupation; kwargs...) for term in basis.terms]
    term_names = [string(nameof(typeof(term))) for term in basis.model.term_types]
    (; energies=Energies(term_names, energy_values))
end

function Hamiltonian(basis::AbstractBasis; ψ=nothing, occupation=nothing, kwargs...)
    energy_hamiltonian(basis, ψ, occupation; kwargs...).ham
end

"""
Get the total local potential of the given Hamiltonian, in real space
in the spin components.
"""
function total_local_potential(ham::Hamiltonian{B}) where {B <: PlaneWaveBasis}
    n_spin = ham.basis.model.n_spin_components
    pots = map(1:n_spin) do σ
        # Get the potential from the first Hamiltonian block of this spin component
        # (works since all local potentials are the same)
        i_σ = first(krange_spin(ham.basis, σ))
        total_local_potential(ham.blocks[i_σ])
    end
    cat(pots..., dims=4)
end
total_local_potential(Hk::DftHamiltonianBlock) = Hk.local_op.potential
function total_local_potential(Hk::GenericHamiltonianBlock)
    only(o for o in Hk.optimized_operators if o isa RealSpaceMultiplication).potential
end
function total_local_potential(ham::Hamiltonian{B}) where {B <: FiniteElementBasis}
    n_spin = ham.basis.model.n_spin_components
    pots = map(1:n_spin) do σ
        # Get the potential from the first Hamiltonian block of this spin component
        # (works since all local potentials are the same)
        i_σ = first(krange_spin(ham.basis, σ))
        total_local_potential(ham.blocks[i_σ])
    end
    cat(pots..., dims=2)
end
function total_local_potential(Hk::GenericHamiltonianBlockFEM)
    only(o for o in Hk.optimized_operators if o isa FEMRealSpaceMultiplication).potential
end

"""
Returns a new Hamiltonian with local potential replaced by the given one
"""
function hamiltonian_with_total_potential(ham::Hamiltonian{B}, V) where {B <: PlaneWaveBasis}
    @assert size(V, 4) == ham.basis.model.n_spin_components
    newblocks = [hamiltonian_with_total_potential(Hk, V[:, :, :, Hk.kpoint.spin])
                 for Hk in ham.blocks]
    Hamiltonian(ham.basis, newblocks)
end
function hamiltonian_with_total_potential(Hk::HamiltonianBlock{B}, V) where {B <: PlaneWaveBasis}
    operators = [op for op in Hk.operators if !(op isa RealSpaceMultiplication)]
    push!(operators, RealSpaceMultiplication(Hk.basis, Hk.kpoint, V))
    HamiltonianBlock(Hk.basis, Hk.kpoint, operators; Hk.scratch)
end
function hamiltonian_with_total_potential(ham::Hamiltonian{B}, V) where {B <: FiniteElementBasis}
    @assert size(V, 2) == ham.basis.model.n_spin_components
    newblocks = [hamiltonian_with_total_potential(Hk, V[:, Hk.kpoint.spin])
                 for Hk in ham.blocks]
    Hamiltonian(ham.basis, newblocks)
end
function hamiltonian_with_total_potential(Hk::HamiltonianBlock{B}, V) where {B <: FiniteElementBasis}
    operators = Vector{FEMOperator}([op for op in Hk.operators if !(op isa FEMRealSpaceMultiplication)])
    push!(operators, FEMRealSpaceMultiplication(Hk.basis, Hk.kpoint, V))
    HamiltonianBlock(Hk.basis, Hk.kpoint, operators; Hk.scratch)
end