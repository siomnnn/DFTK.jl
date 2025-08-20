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
    ψ_dof_handler::DofHandler{3, Grid{3, C, T}}
    ρ_dof_handler::DofHandler{3, Grid{3, C, T}}
    ψ_constraint_handler::ConstraintHandler{DofHandler{3, Grid{3, C, T}}, T}
    ρ_constraint_handler::ConstraintHandler{DofHandler{3, Grid{3, C, T}}, T}
    ψ_cell_values::CellValues
    ρ_cell_values::CellValues
    dof_map::Vector{Int}            # index with ψ dofs to get ρ dofs
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
    ψ_ip = Lagrange{S, degree}()
    ρ_ip = Lagrange{S, 2*degree}()

    ψ_dof_handler, ρ_dof_handler = setup_dofs(grid, ψ_ip, ρ_ip)
    ψ_cell_values, ρ_cell_values = setup_cell_values(ψ_ip, ρ_ip)
    ψ_constraint_handler, ρ_constraint_handler = setup_periodic_boundaries(lattice, ψ_dof_handler, ρ_dof_handler)

    dof_map = setup_dof_map(ψ_dof_handler, ρ_dof_handler, ψ_cell_values)

    return FEMDiscretization{T, S, C}(lattice, grid, ψ_dof_handler, ρ_dof_handler,
                                      ψ_constraint_handler, ρ_constraint_handler,
                                      ψ_cell_values, ρ_cell_values,
                                      dof_map)
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

function setup_dofs(grid::Grid, ψ_ip::Interpolation, ρ_ip::Interpolation)
    ψ_dh = DofHandler(grid)
    add!(ψ_dh, :ψ, ψ_ip)
    close!(ψ_dh)

    ρ_dh = DofHandler(grid)
    add!(ρ_dh, :ρ, ρ_ip)
    close!(ρ_dh)
    return ψ_dh, ρ_dh
end

function setup_cell_values(ψ_ip::Interpolation{S}, ρ_ip::Interpolation{S}; degree=2) where {S <: Ferrite.AbstractRefShape}
    qr = QuadratureRule{S}(degree)
    ψ_cv = CellValues(qr, ψ_ip)
    ρ_cv = CellValues(qr, ρ_ip)
    return ψ_cv, ρ_cv
end

function setup_periodic_boundaries(lattice::Mat3, ψ_dh::DofHandler, ρ_dh::DofHandler)
    periodic_faces = PeriodicFacetPair[]
    for i=1:3
        shift = Tensor{1, 3}(lattice[:, i])
        translation_map(x) = x + shift
        collect_periodic_facets!(periodic_faces, ψ_dh.grid, getfacetset(ψ_dh.grid, "periodic_$(i)a"), getfacetset(ψ_dh.grid, "periodic_$(i)b"), translation_map)
    end

    ψ_ch = ConstraintHandler(ψ_dh)
    periodic_ψ = PeriodicDirichlet(:ψ, periodic_faces)
    add!(ψ_ch, periodic_ψ)
    close!(ψ_ch)
    update!(ψ_ch, 0)

    ρ_ch = ConstraintHandler(ρ_dh)
    periodic_ρ = PeriodicDirichlet(:ρ, periodic_faces)
    add!(ρ_ch, periodic_ρ)
    close!(ρ_ch)
    update!(ρ_ch, 0)
    return ψ_ch, ρ_ch
end

function setup_dof_map(ψ_dh::DofHandler, ρ_dh::DofHandler, ψ_cv::CellValues)
    dof_map = zeros(Int, ndofs(ψ_dh))

    ψ_ref_coords = Ferrite.reference_coordinates(Ferrite.getfieldinterpolation(ψ_dh, Ferrite.find_field(ψ_dh, :ψ)))
    ρ_ref_coords = Ferrite.reference_coordinates(Ferrite.getfieldinterpolation(ρ_dh, Ferrite.find_field(ρ_dh, :ρ)))

    local_map = [findall(x -> x == point, ρ_ref_coords)[1] for point in ψ_ref_coords]

    for cell in CellIterator(ψ_dh)
        reinit!(ψ_cv, cell)
        dof_map[celldofs(cell)] = celldofs(ρ_dh, cell.cellid)[local_map]
    end

    return dof_map
end

function get_dof_handler(disc::FEMDiscretization, field::Symbol)
    if field == :ψ
        return disc.ψ_dof_handler
    elseif field == :ρ
        return disc.ρ_dof_handler
    else
        error("Invalid field: $field. Only :ψ and :ρ are supported.")
    end
end

function get_constraint_handler(disc::FEMDiscretization, field::Symbol)
    if field == :ψ
        return disc.ψ_constraint_handler
    elseif field == :ρ
        return disc.ρ_constraint_handler
    else
        error("Invalid field: $field. Only :ψ and :ρ are supported.")
    end
end

function get_cell_values(disc::FEMDiscretization, field::Symbol)
    if field == :ψ
        return disc.ψ_cell_values
    elseif field == :ρ
        return disc.ρ_cell_values
    else
        error("Invalid field: $field. Only :ψ and :ρ are supported.")
    end
end

get_dof_map(disc::FEMDiscretization) = disc.dof_map