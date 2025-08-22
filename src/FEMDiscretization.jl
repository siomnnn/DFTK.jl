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
    ψ_inverse_constraint_map::Vector{Int}       # map from prescribed dofs to free dofs
    ρ_inverse_constraint_map::Vector{Int}
    dof_map::Vector{Int}                        # index with ψ dofs to get ρ dofs
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

    ψ_dof_handler, ρ_dof_handler = setup_dofs(grid, ψ_ip, :ψ), setup_dofs(grid, ρ_ip, :ρ)
    ψ_cell_values, ρ_cell_values = setup_cell_values(ψ_ip), setup_cell_values(ρ_ip)
    ψ_constraint_handler, ρ_constraint_handler = setup_periodic_boundaries(lattice, ψ_dof_handler, ρ_dof_handler)

    ψ_inverse_constraint_map, ρ_inverse_constraint_map = setup_inverse_constraint_map(ψ_constraint_handler), setup_inverse_constraint_map(ρ_constraint_handler)

    dof_map = setup_dof_map(ψ_dof_handler, ρ_dof_handler, ψ_cell_values, ψ_inverse_constraint_map, ρ_inverse_constraint_map)

    return FEMDiscretization{T, S, C}(lattice, grid, ψ_dof_handler, ρ_dof_handler,
                                      ψ_constraint_handler, ρ_constraint_handler,
                                      ψ_cell_values, ρ_cell_values,
                                      ψ_inverse_constraint_map, ρ_inverse_constraint_map,
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

function setup_dofs(grid::Grid, ip::Interpolation, field::Symbol)
    dh = DofHandler(grid)
    add!(dh, field, ip)
    close!(dh)
    return dh
end

function setup_cell_values(ip::Interpolation{S}; degree=2) where {S <: Ferrite.AbstractRefShape}
    qr = QuadratureRule{S}(degree)
    cv = CellValues(qr, ip)
    return cv
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

function setup_inverse_constraint_map(ch::ConstraintHandler)
    inverse_constraint_map = zeros(Int, ndofs(ch.dh))
    for (i, dof) in enumerate(ch.free_dofs)
        inverse_constraint_map[dof] = i
    end
    for (i, pdof) in pairs(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        if dofcoef !== nothing
            @assert length(dofcoef) == 1 "General affine constraints are not supported."
            inverse_constraint_map[pdof] = inverse_constraint_map[dofcoef[1][1]]
        end
    end
    return inverse_constraint_map
end

function setup_dof_map(ψ_dh::DofHandler, ρ_dh::DofHandler, ψ_cv::CellValues, ψ_inverse_constraint_map::Vector{Int}, ρ_inverse_constraint_map::Vector{Int})
    dof_map = zeros(Int, maximum(ψ_inverse_constraint_map))

    ψ_ref_coords = Ferrite.reference_coordinates(Ferrite.getfieldinterpolation(ψ_dh, Ferrite.find_field(ψ_dh, :ψ)))
    ρ_ref_coords = Ferrite.reference_coordinates(Ferrite.getfieldinterpolation(ρ_dh, Ferrite.find_field(ρ_dh, :ρ)))

    local_map = [findall(x -> x == point, ρ_ref_coords)[1] for point in ψ_ref_coords]

    for cell in CellIterator(ψ_dh)
        reinit!(ψ_cv, cell)

        periodic_cell_dofs_ψ = ψ_inverse_constraint_map[celldofs(cell)]
        periodic_cell_dofs_ρ = ρ_inverse_constraint_map[celldofs(ρ_dh, cell.cellid)]

        dof_map[periodic_cell_dofs_ψ] = periodic_cell_dofs_ρ[local_map]
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

function get_inverse_constraint_map(disc::FEMDiscretization, field::Symbol)
    if field == :ψ
        return disc.ψ_inverse_constraint_map
    elseif field == :ρ
        return disc.ρ_inverse_constraint_map
    else
        error("Invalid field: $field. Only :ψ and :ρ are supported.")
    end
end

get_n_dofs(disc::FEMDiscretization, field::Symbol) = ndofs(get_dof_handler(disc, field))
get_n_free_dofs(disc::FEMDiscretization, field::Symbol) = length(get_constraint_handler(disc, field).free_dofs)
get_free_dofs(disc::FEMDiscretization, field::Symbol) = get_constraint_handler(disc, field).free_dofs

get_dof_map(disc::FEMDiscretization) = disc.dof_map
reduce_dofs(disc::FEMDiscretization, f) = f[get_dof_map(disc)]

apply_inverse_constraint_map(disc::FEMDiscretization, f, field::Symbol) = get_inverse_constraint_map(disc, field)[f]

function init_overlap_matrix(disc::FEMDiscretization{T}, field::Symbol) where T
    dh = get_dof_handler(disc, field)
    ch = get_constraint_handler(disc, field)
    cv = get_cell_values(disc, field)

    H = allocate_matrix(dh, ch)
    assembler = start_assemble(H)

    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(complex(T), n_basefuncs, n_basefuncs)

    n_quad = getnquadpoints(cv)
    ϕ_evals = shape_value.([cv], 1:n_quad, (1:n_basefuncs)')

    # TODO: is parallelization possible even though we are reinit-ing cell_values?
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        
        dΩ = getdetJdV.([cv], 1:n_quad)
        
        for i in 1:n_basefuncs, j in 1:n_basefuncs
            Ke[i, j] += (ϕ_evals[:, i] .* ϕ_evals[:, j])' * dΩ
        end
    
        assemble!(assembler, celldofs(cell), Ke)
    end

    Ferrite.apply!(H, ch)

    free_dofs = get_free_dofs(disc, field)
    H[free_dofs, free_dofs]
end

function init_neg_half_laplace_matrix(disc::FEMDiscretization{T}, field) where T
    dh = get_dof_handler(disc, field)
    ch = get_constraint_handler(disc, field)
    cv = get_cell_values(disc, field)

    neg_half_laplace = allocate_matrix(dh, ch)

    n_basefuncs = getnbasefunctions(cv)
    Ke = zeros(complex(T), n_basefuncs, n_basefuncs)
    
    assembler = start_assemble(neg_half_laplace)

    n_quad = getnquadpoints(cv)

    # TODO: is parallelization possible even though we are reinit-ing cell_values?
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        fill!(Ke, 0)
        
        ∇ϕ_evals = shape_gradient.([cv], 1:n_quad, (1:n_basefuncs)')
        dΩ = getdetJdV.([cv], 1:n_quad)
        
        for i in 1:n_basefuncs, j in 1:n_basefuncs
            Ke[i, j] += 0.5 * (∇ϕ_evals[:, i] .⋅ ∇ϕ_evals[:, j])' * dΩ
        end
    
        assemble!(assembler, celldofs(cell), Ke)
    end

    Ferrite.apply!(neg_half_laplace, ch)

    free_dofs = get_free_dofs(disc, field)
    neg_half_laplace[free_dofs, free_dofs]
end

function get_dof_positions(disc::FEMDiscretization{T}, field::Symbol) where T
    dh = get_dof_handler(disc, field)
    ip = Ferrite.getfieldinterpolation(dh, Ferrite.find_field(dh, field))
    ref_coords = hcat(Ferrite.reference_coordinates(ip)...)
    cell_values = get_cell_values(disc, field)

    dof_coords = zeros(SVector{3, T}, get_n_free_dofs(disc, field))
    for cell in CellIterator(dh)
        reinit!(cell_values, cell)
        
        cell_dof_coords = (cell.coords[1] * (1 .- ref_coords[1, :] .- ref_coords[2, :] .- ref_coords[3, :])'
                            .+ cell.coords[2] * ref_coords[1, :]'
                            .+ cell.coords[3] * ref_coords[2, :]'
                            .+ cell.coords[4] * ref_coords[3, :]')
        dof_coords[apply_inverse_constraint_map(disc, celldofs(cell), field)] .= SVector{3}.(eachcol(cell_dof_coords))
    end
    return dof_coords
end

# The way Ferrite applies boundary conditions to vectors is not the way that we want it to.
# In particular, it does not "merge" the values of periodic dofs, but rather overwrites one with
# the other. Here, we need to consider all periodic dofs _not_ as redundant/overwritable,
# but rather as components of the same coefficient -> overwrite them with their sum.
function apply_bc!(f::AbstractVector, ch::ConstraintHandler)
    for (i, pdof) in pairs(ch.prescribed_dofs)
        dofcoef = ch.dofcoefficients[i]
        if dofcoef !== nothing # if affine constraint
            for (d, v) in dofcoef
                f[d] += f[pdof] * v
            end
        end
        f[pdof] = 0
    end
    return
end

remove_bc!(f::AbstractVector, ch::ConstraintHandler) = (f[ch.prescribed_dofs] .= 0)