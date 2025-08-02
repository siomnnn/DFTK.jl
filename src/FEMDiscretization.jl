using Ferrite

@doc raw"""
A `FEMDiscretization` consists of the lattice discretized by a finite element
`Grid` together information about the basis functions, periodicity constraints,
and relevant quadrature rules.

This structure only contains spatial information and nothing about the physical
model (beyond the size of the unit cell).
"""
struct FEMDiscretization{T}
    lattice::Mat3{T}
    grid::Grid{T}
    dof_handler::DofHandler{T}
    constraint_handler::ConstraintHandler{T}
    cell_values::CellValues{T}
end

@doc raw"""
Constructs a `FEMDiscretization` from a lattice and a Ferrite `Grid` object.
The optional `degree` argument specifies the polynomial degree of the finite
element basis functions; if none is provided, it defaults to linear elements.
"""
function FEMDiscretization(lattice::Mat3{T}, grid::Grid{T}; degree::Int=1) where T
    φ_ip = Lagrange{RefTetrahedron, degree}()

    dof_handler = setup_dofs(grid, φ_ip)
    cell_values = setup_cell_values(φ_ip)

    constraint_handler = setup_periodic_boundaries(lattice, dof_handler)
    return FEMDiscretization{T}(lattice, grid, dof_handler, constraint_handler, cell_values)
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

function setup_cell_values(φ_ip::Interpolation)
    qr = QuadratureRule{RefTetrahedron}(2)
    φ_cv = CellValues(qr, φ_ip)
    return φ_cv
end

function setup_periodic_boundaries(lattice::Mat3{T}, dh::DofHandler{T}) where T
    ch = ConstraintHandler(dh)

    for i=1:3
        translation_map(x) = x + lattice[:, i]
        periodic_faces = collect_periodic_facets(dh.grid, "periodic_$(i)a", "periodic_$(i)b", translation_map)
        periodic = PeriodicDirichlet(:φ, periodic_faces, [1, 2])
        add!(ch, periodic)
    end

    close!(ch)
    update!(ch, 0)
    return ch
end