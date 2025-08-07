using Ferrite
using FerriteGmsh
using Gmsh: gmsh

@doc raw"""
A finite-element discretized `Model`.
"""
struct FiniteElementBasis{T,
                      VT <: Real,
                      Arch <: AbstractArchitecture,
                     } <: AbstractBasis{T}

    # T is the default type to express data, VT the corresponding bare value type (i.e. not dual)
    model::Model{T, VT}

    ## FEM basis information
    h::T  # Target grid spacing in real space.
    degree::Int  # Polynomial degree of the finite element basis functions.
    discretization::FEMDiscretization{T}  # Real-space finite element discretization of the unit cell.

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
                            architecture::Arch,
                           ) where {T, VT, Arch}
    terms = Vector{Any}(undef, length(model.term_types))  # Dummy terms array, filled below

    basis = FiniteElementBasis{T, VT, Arch}(model, austrip(h), degree, discretization, architecture, terms)

    # TODO: make terms work
    #for (it, t) in enumerate(model.term_types)
    #    term_name = string(nameof(typeof(t)))
    #    @timing "Instantiation $term_name" basis.terms[it] = t(basis)
    #end
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
so this function is not compatible with arbitrary meshes.

Only 3D meshes are currently supported.
"""
@timing function FiniteElementBasis(model::Model{T};
                                h::T,
                                degree::Int=1,
                                grid=nothing,
                                write_to_file=false,
                                filename="mesh.msh",
                                architecture=CPU()
                               ) where {T <: Real}
    @assert grid isa Union{Nothing, Grid} "grid must be a Ferrite Grid or nothing"

    if isnothing(grid)
        grid = construct_FEM_grid(model, austrip(h); write_to_file=write_to_file, filename=filename)
    end

    discretization = FEMDiscretization(model.lattice, grid; degree)

    FiniteElementBasis(model, austrip(h), degree, discretization, architecture)
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

getndofs(basis::FiniteElementBasis) = basis.discretization.dof_handler.ndofs
getdofhandler(basis::FiniteElementBasis) = basis.discretization.dof_handler
getconstrainthandler(basis::FiniteElementBasis) = basis.discretization.constraint_handler
getcellvalues(basis::FiniteElementBasis) = basis.discretization.cell_values