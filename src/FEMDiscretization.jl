using Ferrite
using Tensors

@doc raw"""
A `FEMDiscretization` consists of the unit cell discretized by a finite element
`Grid` together with information about the basis functions, periodicity constraints,
and relevant quadrature rules.

This structure only contains spatial information and nothing about the physical
model (beyond the size of the unit cell).
"""
struct FEMDiscretization{T, S <: Ferrite.AbstractRefShape, C <: Ferrite.AbstractCell{S}}
    lattice::Mat3{T}
    grid::Grid{3, C, T}
    dof_handler::DofHandler{3, Grid{3, C, T}}
    constraint_handler::ConstraintHandler{DofHandler{3, Grid{3, C, T}}, T}
    cell_values::CellValues
end

@doc raw"""
Constructs a `FEMDiscretization` from a lattice and a Ferrite `Grid` object.
The optional `degree` argument specifies the polynomial degree of the finite
element basis functions; if none is provided, it defaults to linear elements.
"""
function FEMDiscretization(lattice::Mat3{T},
                           grid::Grid{3, C, T};
                           degree::Int=1
                          ) where {T, S <: Ferrite.AbstractRefShape, C <: Ferrite.AbstractCell{S}}
    φ_ip = Lagrange{S, degree}()

    dof_handler = setup_dofs(grid, φ_ip)
    cell_values = setup_cell_values(φ_ip)

    constraint_handler = setup_periodic_boundaries(lattice, dof_handler)
    return FEMDiscretization{T, S, C}(lattice, grid, dof_handler, constraint_handler, cell_values)
end

@doc raw"""
Constructs a `FEMDiscretization` from a lattice and a mesh file that was
pre-generated using `construct_FEM_grid`. The optional `degree` argument
specifies the polynomial degree of the finite element basis functions; if none
is provided, it defaults to linear elements.
"""
function FEMDiscretization(lattice::Mat3{T}, filename::String; degree::Int=1) where T
    grid = togrid(filename)
    return FEMDiscretization(lattice, grid; degree)
end

function setup_dofs(grid::Grid, φ_ip::Interpolation)
    dh = DofHandler(grid)

    add!(dh, :φ, φ_ip)

    close!(dh)
    return dh
end

function setup_cell_values(φ_ip::Interpolation{S}; degree=2) where {S <: Ferrite.AbstractRefShape}
    qr = QuadratureRule{S}(degree)
    φ_cv = CellValues(qr, φ_ip)
    return φ_cv
end

function setup_periodic_boundaries(lattice::Mat3{T}, dh::DofHandler) where T
    ch = ConstraintHandler(dh)

    periodic_faces = PeriodicFacetPair[]
    for i=1:3
        shift = Tensor{1, 3}(lattice[:, i])
        translation_map(x) = x + shift
        collect_periodic_facets!(periodic_faces, dh.grid, getfacetset(dh.grid, "periodic_$(i)a"), getfacetset(dh.grid, "periodic_$(i)b"), translation_map)
    end
    periodic = PeriodicDirichlet(:φ, periodic_faces)
    add!(ch, periodic)

    close!(ch)
    update!(ch, 0)
    return ch
end