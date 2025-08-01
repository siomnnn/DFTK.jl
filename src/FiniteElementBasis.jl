using Ferrite
using FerriteGmsh
using Gmsh: gmsh

@doc raw"""
A finite-element discretized `Model`.
Normalization conventions:
- Quantities expressed on the real-space grid are in actual values.
"""
struct FiniteElementBasis{T,
                      VT <: Real,
                      Arch <: AbstractArchitecture,
                     } <: AbstractBasis{T}

    # T is the default type to express data, VT the corresponding bare value type (i.e. not dual)
    model::Model{T, VT}

    ## FEM basis information
    h::T  # Target grid spacing in real space.
    grid::Grid  # Real-space grid from Ferrite.

    ## Information on the hardware and device used for computations.
    architecture::Arch

    ## Instantiated terms (<: Term). See Hamiltonian for high-level usage
    terms::Vector{Any}
end

# Lowest-level constructor. Only call if you know what you're doing.
function FiniteElementBasis(model::Model{T, VT},
                            h::T,
                            grid::Grid,
                            architecture::Arch,
                           ) where {T, VT, Arch}
    terms = Vector{Any}(undef, length(model.term_types))  # Dummy terms array, filled below

    basis = FiniteElementBasis{T, VT, Arch}(model, grid, h, architecture, terms)

    for (it, t) in enumerate(model.term_types)
        term_name = string(nameof(typeof(t)))
        @timing "Instantiation $term_name" basis.terms[it] = t(basis)
    end
    basis
end

@doc raw"""
Creates a `FiniteElementBasis` using the mesh width `h` and a Ferrite `Grid` object.

If `grid` is not provided, a grid is constructed using the `construct_FEM_grid` function.
This is slow if if many calculations are done using the same model and mesh width. In order
to speed things up, it is recommended to either pre-generate the grid using `construct_FEM_grid`
or by setting `keep_grid=true` to save the grid to the file in `filename`. Then, load the
saved grid using the alternative constructor.
"""
@timing function PlaneWaveBasis(model::Model{T};
                                h::T,
                                grid=nothing,
                                write_to_file=false,
                                filename="mesh.msh",
                                architecture=CPU()
                               ) where {T <: Real}
    @assert grid isa Union{Nothing, Grid} "grid must be a Ferrite Grid or nothing"

    if isnothing(grid)
        grid = construct_FEM_grid(model, austrip(h); write_to_file=write_to_file, filename=filename)
    end

    FiniteElementBasis(model, austrip(h), grid, architecture)
end

@doc raw"""
Creates a `FiniteElementBasis` using the mesh width `h` and a `.msh` file located at `filename`.

This function is preferred when a grid has already been generated and saved to a file.
It is faster than the other constructor, as it does not need to generate the grid from scratch.
"""
@timing function PlaneWaveBasis(model::Model{T};
                                h::T,
                                filename::String,
                                architecture=CPU()
                               ) where {T <: Real}
    grid = FerriteGmsh.togrid(filename, domain="")

    FiniteElementBasis(model, austrip(h), grid, architecture)
end

# prevent broadcast
Base.Broadcast.broadcastable(basis::FiniteElementBasis) = Ref(basis)

Base.eltype(::FiniteElementBasis{T}) where {T} = T

@doc raw"""
Constructs a Ferrite grid object from a `Model` and a mesh width `h`.
Can be saved in Gmsh `.msh` format for later use by setting `write_to_file=true` and providing a `filename`.
"""
function construct_FEM_grid(model::Model{T}, h::T; write_to_file=false, filename="mesh.msh") where T
    # TODO: this is incredibly ugly, find a better way to do this
    lattice = model.lattice
    if size(lattice, 2) != 3
        error("Lattice must be 3-dimensional.")
    end

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal",0)

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
    gmsh.model.geo.extrude([2, 10], lattice[:, 3]...)
    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], 1, "unit_cell")

    # Specify periodicity
    translation = [1, 0, 0, lattice[1, 3],
                   0, 1, 0, lattice[2, 3],
                   0, 0, 1, lattice[3, 3],
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [32], [10], translation)
    translation = [1, 0, 0, lattice[1, 2],
                   0, 1, 0, lattice[2, 2],
                   0, 0, 1, lattice[3, 2],
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [27], [19], translation)
    translation = [1, 0, 0, lattice[1, 1],
                   0, 1, 0, lattice[2, 1],
                   0, 0, 1, lattice[3, 1],
                   0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(2, [23], [31], translation)

    # Generate mesh
    gmsh.model.mesh.generate(3)
    
    if write_to_file
        gmsh.write(filename)
    end

    # Convert to Ferrite Grid
    grid = FerriteGmsh.togrid(filename, domain="unit_cell")

    gmsh.finalize()

    return grid
end