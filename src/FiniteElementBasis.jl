using Ferrite
using FerriteGmsh
using Gmsh: gmsh
using Krylov

# temporary, rudimentary implementation of k-points for FEM
struct FEMKpoint{T <: Real}
    spin::Int
    coordinate::Vec3{T}
end

@doc raw"""
A finite-element discretized `Model`.
"""
struct FiniteElementBasis{T,
                      VT <: Real,
                      NFFTtype <: NFFTGrid{T, VT},
                      Arch <: AbstractArchitecture,
                     } <: AbstractBasis{T}

    # T is the default type to express data, VT the corresponding bare value type (i.e. not dual)
    model::Model{T, VT}

    ## FEM basis information
    h::T  # Target grid spacing in real space.
    degree::Int  # Polynomial degree of the finite element basis functions.
    discretization::FEMDiscretization{T}  # Real-space finite element discretization of the unit cell.

    ψ_overlap_matrix::AbstractMatrix{T}
    ψ_neg_half_laplacian::Union{AbstractMatrix{T}, Nothing}

    ρ_overlap_matrix::AbstractMatrix{T}
    ρ_neg_half_laplacian::Union{AbstractMatrix{T}, Nothing}

    refinement_matrix::AbstractMatrix{T}        # Matrix to refine a function from the coarser ψ interpolation to the finer ρ interpolation (ψ_fine = refinement_matrix * ψ_coarse).
    
    nfft_size::Tuple{Int, Int, Int}
    nfft_grid::NFFTtype

    kpoints::Vector{FEMKpoint{T}}
    kweights::Vector{T}

    ## Information on the hardware and device used for computations.
    architecture::Arch

    ## Instantiated terms (<: Term). See Hamiltonian for high-level usage
    terms::Vector{Any}
end

# Lowest-level constructor. Only call if you know what you're doing.
function FiniteElementBasis(model::Model{T, VT},
                            h::T,
                            degree::Int,
                            discretization::FEMDiscretization{T},
                            nfft_size::Tuple{Int, Int, Int},
                            kpoints::Vector{FEMKpoint{T}},
                            kweights::Vector{T},
                            precompute_laplacian::Bool,
                            architecture::Arch,
                           ) where {T, VT, Arch}
    @assert length(kpoints) == length(kweights) "kpoints and kweights must have the same length"
    terms = Vector{Any}(undef, length(model.term_types))  # Dummy terms array, filled below

    ψ_overlap_matrix = init_overlap_matrix(discretization, :ψ)
    ρ_overlap_matrix = init_overlap_matrix(discretization, :ρ)

    ψ_neg_half_laplacian = nothing
    ρ_neg_half_laplacian = nothing
    if precompute_laplacian
        ψ_neg_half_laplacian = init_neg_half_laplace_matrix(discretization, :ψ)
        ρ_neg_half_laplacian = init_neg_half_laplace_matrix(discretization, :ρ)
    end
    refinement_matrix = init_refinement_matrix(discretization)

    # NFFT only likes nodes in [-0.5, 0.5)
    reduced_dof_coords = ([model.lattice] .\ get_dof_positions(discretization, :ρ)) .- [Vec3{VT}(0.5, 0.5, 0.5)]
    for i in eachindex(reduced_dof_coords)
        mask = reduced_dof_coords[i] .>= 0.5
        neg_mask = reduced_dof_coords[i] .< -0.5
        reduced_dof_coords[i] += (neg_mask - mask) .* Vec3{VT}(1.0, 1.0, 1.0)
    end
    nfft_grid = NFFTGrid(nfft_size, reduced_dof_coords, model.unit_cell_volume, architecture)

    basis = FiniteElementBasis{T, VT, typeof(nfft_grid), Arch}(model, austrip(h), degree, discretization,
                                           ψ_overlap_matrix, ψ_neg_half_laplacian,
                                           ρ_overlap_matrix, ρ_neg_half_laplacian,
                                           refinement_matrix, nfft_size, nfft_grid,
                                           kpoints, kweights,
                                           architecture, terms)

    for (it, t) in enumerate(model.term_types)
        term_name = string(nameof(typeof(t)))
        @timing "Instantiation $term_name" basis.terms[it] = t(basis)
    end
    basis
end

@doc raw"""
Creates a `FiniteElementBasis` using the mesh width `h` and a Ferrite `Grid` object.

If `grid` is not provided, a grid is constructed using the `construct_FEM_grid` function.
This is unnecessarily slow if many calculations are done using the same model and mesh width.
In order to speed things up, it is recommended to pre-generate the grid either using
`construct_FEM_grid` or by setting `write_to_file=true` to save the grid to `filename`
(by default `mesh.msh`). Then, load the saved grid using `load_grid_from_file`.

Note that such a mesh file requires a rather specific labeling of the periodic faces,
so this function is not compatible with arbitrary meshes. Only 3D meshes are currently supported.

By default, the matrix representation of the Laplacian operator is precomputed. This can be
disabled by setting `precompute_laplacian=false`.
"""
@timing function FiniteElementBasis(model::Model{T};
                                    h::Number,
                                    degree::Int=1,
                                    grid=nothing,
                                    nfft_size=nothing,
                                    kpoints::Vector{FEMKpoint{T}}=[FEMKpoint(1, Vec3{T}(0, 0, 0))],
                                    kweights::Vector{T}=[one(T)],
                                    precompute_laplacian=true,
                                    architecture=CPU(),
                                    write_to_file=false,
                                    filename="mesh.msh"
                                   ) where {T <: Real}
    @assert grid isa Union{Nothing, Grid} "grid must be a Ferrite Grid or nothing"

    if isnothing(grid)
        grid = construct_FEM_grid(model, austrip(h); write_to_file=write_to_file, filename=filename)
    end
    if isnothing(nfft_size)
        nfft_size = tuple(2 .^ ceil.(Int, log2.(norm.(eachcol(model.lattice)) ./ austrip(h)))...)
    end

    discretization = FEMDiscretization(model.lattice, grid; degree)

    FiniteElementBasis(model, austrip(h), degree, discretization, nfft_size, kpoints, kweights, precompute_laplacian, architecture)
end

# prevent broadcast
Base.Broadcast.broadcastable(basis::FiniteElementBasis) = Ref(basis)

Base.eltype(::FiniteElementBasis{T}) where {T} = T

@doc raw"""
Constructs a Ferrite grid object from a `Model` and a mesh width `h`. Can be saved
in Gmsh `.msh` format for later use by setting `write_to_file=true` and providing a `filename`.
"""
function construct_FEM_grid(model::Model{T}, h::T; write_to_file=false, filename="mesh.msh") where T
    # TODO: this is incredibly ugly, find a better way to do this
    lattice = model.lattice
    h = austrip(h)

    gmsh.initialize()

    # Suppress terminal output
    gmsh.option.setNumber("General.Terminal",0)

    # multithreading
    gmsh.option.setNumber("General.NumThreads",DFTK_threads.x)

    gmsh.model.add("unit_cell")

    gmsh.model.geo.addPoint([0, 0, 0]..., h, 1)
    gmsh.model.geo.addPoint(lattice[:, 1]..., h, 2)
    gmsh.model.geo.addPoint((lattice[:, 1] + lattice[:, 2])..., h, 3)
    gmsh.model.geo.addPoint(lattice[:, 2]..., h, 4)

    gmsh.model.geo.addLine(1, 2, 5)
    gmsh.model.geo.addLine(2, 3, 6)
    gmsh.model.geo.addLine(3, 4, 7)
    gmsh.model.geo.addLine(4, 1, 8)

    gmsh.model.geo.addCurveLoop([5, 6, 7, 8], 9)
    gmsh.model.geo.addPlaneSurface([9], 10)

    # Extrude to volume
    gmsh.model.geo.extrude([(2, 10)], lattice[:, 3]...)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], 1, "unit_cell")

    # Label periodic faces
    gmsh.model.addPhysicalGroup(2, [23], -1, "periodic_1a")
    gmsh.model.addPhysicalGroup(2, [31], -1, "periodic_1b")
    gmsh.model.addPhysicalGroup(2, [27], -1, "periodic_2a")
    gmsh.model.addPhysicalGroup(2, [19], -1, "periodic_2b")
    gmsh.model.addPhysicalGroup(2, [32], -1, "periodic_3a")
    gmsh.model.addPhysicalGroup(2, [10], -1, "periodic_3b")

    # Specify mesh periodicity
    translation = [1, 0, 0, lattice[1, 1],
                   0, 1, 0, lattice[2, 1],
                   0, 0, 1, lattice[3, 1],
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [23], [31], translation)
    translation = [1, 0, 0, lattice[1, 2],
                   0, 1, 0, lattice[2, 2],
                   0, 0, 1, lattice[3, 2],
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [27], [19], translation)
    translation = [1, 0, 0, lattice[1, 3],
                   0, 1, 0, lattice[2, 3],
                   0, 0, 1, lattice[3, 3],
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [32], [10], translation)

    # Generate mesh
    gmsh.model.mesh.generate(3)
    
    # Convert to Ferrite Grid. The temporary file in the else part is necessary
    # due to a FerriteGmsh bug which messes up the boundary tags when converting
    # directly from the current Gmsh state (even though there is a function for that).
    if write_to_file
        gmsh.write(filename)
        grid = togrid(filename; domain="unit_cell")
    else
        grid = mktempdir() do dir
            path = joinpath(dir, "mesh.msh")
            gmsh.write(path)
            togrid(path; domain="unit_cell")
        end
    end

    gmsh.finalize()

    return grid
end

function load_grid_from_file(filename::String)
    grid = togrid(filename; domain="unit_cell")
    return grid
end

get_n_dofs(basis::FiniteElementBasis) = (; ψ=basis.discretization.ψ_dof_handler.ndofs, ρ=basis.discretization.ρ_dof_handler.ndofs)
get_n_free_dofs(basis::FiniteElementBasis) = (; ψ=length(basis.discretization.ψ_constraint_handler.free_dofs), ρ=length(basis.discretization.ρ_constraint_handler.free_dofs))
get_dof_handlers(basis::FiniteElementBasis) = (; ψ=basis.discretization.ψ_dof_handler, ρ=basis.discretization.ρ_dof_handler)
get_constraint_handlers(basis::FiniteElementBasis) = (; ψ=basis.discretization.ψ_constraint_handler, ρ=basis.discretization.ρ_constraint_handler)
get_cell_values(basis::FiniteElementBasis) = (; ψ=basis.discretization.ψ_cell_valuescell_values, ρ=basis.discretization.ρ_cell_values)
get_nodes(basis::FiniteElementBasis) = basis.discretization.grid.nodes

get_n_dofs(basis::FiniteElementBasis, field::Symbol) = get_n_dofs(basis.discretization, field)
get_n_free_dofs(basis::FiniteElementBasis, field::Symbol) = get_n_free_dofs(basis.discretization, field)
get_free_dofs(basis::FiniteElementBasis, field::Symbol) = get_free_dofs(basis.discretization, field)
get_dof_handler(basis::FiniteElementBasis, field::Symbol) = get_dof_handler(basis.discretization, field)
get_periodic_dofset(basis::FiniteElementBasis, field::Symbol) = get_periodic_dofset(basis.discretization, field)
get_constraint_handler(basis::FiniteElementBasis, field::Symbol) = get_constraint_handler(basis.discretization, field)
get_cell_values(basis::FiniteElementBasis, field::Symbol) = get_cell_values(basis.discretization, field)
get_inverse_constraint_map(basis::FiniteElementBasis, field::Symbol) = get_inverse_constraint_map(basis.discretization, field)

get_dof_positions(basis::FiniteElementBasis, field::Symbol) = get_dof_positions(basis.discretization, field)
get_dof_map(basis::FiniteElementBasis) = get_dof_map(basis.discretization)
reduce_dofs(basis::FiniteElementBasis, f) = f[get_dof_map(basis)]

apply_inverse_constraint_map(basis::FiniteElementBasis, f, field::Symbol) = get_inverse_constraint_map(basis, field)[f]

function get_neg_half_laplace_matrix(basis::FiniteElementBasis, field::Symbol)
    if field == :ψ
        return basis.ψ_neg_half_laplacian
    elseif field == :ρ
        return basis.ρ_neg_half_laplacian
    else
        error("Invalid field: $field. Only :ψ and :ρ are supported.")
    end
end

function get_constraint_matrix(basis::FiniteElementBasis{T}, kpoint::FEMKpoint{T}, field::Symbol) where {T}
    lattice = basis.model.lattice
    k = kpoint.coordinate

    ch = get_constraint_handler(basis, field)
    
    I, J, vals = Int[], Int[], Complex{T}[]

    for (j, d) in enumerate(ch.free_dofs)
        push!(I, d)
        push!(J, j)
        push!(vals, 1.0)
    end

    periodic_dofset = get_periodic_dofset(basis.discretization, field)

    for (i, pdof) in enumerate(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        if dofcoef !== nothing
            @assert length(dofcoef) == 1
            (d, _) = dofcoef[1]
            push!(I, pdof)
            j = searchsortedfirst(ch.free_dofs, d)
            push!(J, j)

            lattice_dir = [pdof in periodic_dofset[i] for i in 1:3]
            phase = exp(im * dot(k, lattice * lattice_dir))
            push!(vals, phase)
        end
    end

    return SparseArrays.sparse!(I, J, vals, get_n_dofs(basis, field), get_n_free_dofs(basis, field))
end

function get_overlap_matrix(basis::FiniteElementBasis, field::Symbol)
    if field == :ψ
        return basis.ψ_overlap_matrix
    elseif field == :ρ
        return basis.ρ_overlap_matrix
    else
        error("Invalid field: $field. Only :ψ and :ρ are supported.")
    end
end

get_refinement_matrix(basis::FiniteElementBasis) = basis.refinement_matrix

LinearAlgebra.norm(ψ::AbstractVector{T}, basis::FiniteElementBasis{T}, field::Symbol) where T = dot(ψ, get_overlap_matrix(basis, field), ψ)^0.5
LinearAlgebra.norm(ψ::AbstractMatrix{T}, basis::FiniteElementBasis{T}, field::Symbol) where T = sum(dot(ψn, get_overlap_matrix(basis, field), ψn) for ψn in eachcol(ψ))^0.5

integrate(f::AbstractVector{T}, basis::FiniteElementBasis{T}, field::Symbol) where T = dot(ones(T, length(f)), get_overlap_matrix(basis, field), f)

function solve_laplace(basis::FiniteElementBasis{T}, f::AbstractVector{T}, field::Symbol) where T
    constraint_matrix = get_constraint_matrix(basis, FEMKpoint(1, Vec3{T}(0, 0, 0)), field)
    mat = constraint_matrix' * get_neg_half_laplace_matrix(basis, field) * constraint_matrix
    rhs = complex.(get_overlap_matrix(basis, :ρ) * f)
    if !isnothing(mat)
        (x, stats) = minres_qlp(mat, rhs)
        stats.solved || error("Laplacian solve did not converge")
        return real.(x)
    end

    op = NegHalfLaplaceFEMOperator(basis)
    (x, stats) = minres_qlp(op, rhs)
    stats.solved || error("Laplacian solve did not converge")
    return real.(x)
end

# not assuming that kpoints are sorted by spin, since they are user-specified.
function krange_spin(basis::FiniteElementBasis, spin::Integer)
    findall(kpt -> kpt.spin == spin, basis.kpoints)
end

function weighted_ksum(basis::FiniteElementBasis, array)
    sum(basis.kweights .* array)
end

G_vectors(basis::FiniteElementBasis) = G_vectors(basis.nfft_size)

"""
Forward NFFT calls to the FiniteElementBasis nfft_grid field
"""
nfft(basis::FiniteElementBasis, f_real::AbstractArray3) = 
    nfft(basis.nfft_grid, f_real)
nfft!(f_fourier::AbstractArray3, basis::FiniteElementBasis, f_real::AbstractArray3) = 
    nfft!(f_fourier, basis.nfft_grid, f_real)
anfft(basis::FiniteElementBasis, f_real::AbstractArray3) = 
    anfft(basis.nfft_grid, f_real)
anfft!(f_real::AbstractArray3, basis::FiniteElementBasis, f_fourier::AbstractArray3) = 
    nfft!(f_real, basis.nfft_grid, f_fourier)
rnfft(basis::FiniteElementBasis, f_real::AbstractArray3) = 
    rnfft(basis.nfft_grid, f_real)
ranfft(basis::FiniteElementBasis, f_fourier::AbstractArray3) = 
    ranfft(basis.nfft_grid, f_fourier)
