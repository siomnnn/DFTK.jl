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
    ψ_per_dofset::Vector{Vector{Int}}
    ρ_per_dofset::Vector{Vector{Int}}
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
    ψ_per_dofset, ρ_per_dofset = setup_periodic_dofsets(ψ_dof_handler), setup_periodic_dofsets(ρ_dof_handler)
    ψ_constraint_handler, ρ_constraint_handler = setup_periodic_boundaries(lattice, ψ_dof_handler, ρ_dof_handler)

    ψ_inverse_constraint_map, ρ_inverse_constraint_map = setup_inverse_constraint_map(ψ_constraint_handler), setup_inverse_constraint_map(ρ_constraint_handler)

    dof_map = setup_dof_map(ψ_dof_handler, ρ_dof_handler, ψ_cell_values, ψ_inverse_constraint_map, ρ_inverse_constraint_map)

    return FEMDiscretization{T, S, C}(lattice, grid, ψ_dof_handler, ρ_dof_handler,
                                      ψ_per_dofset, ρ_per_dofset,
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

function get_dofs_on_facetset(dh::DofHandler, facetset::String)
    dofs = Int[]
    field_name = dh.field_names[1]
    field_ip = Ferrite.getfieldinterpolation(dh, Ferrite.find_field(dh, field_name))

    for facet_id in getfacetset(dh.grid, facetset)
        cell_id, local_facet_id = facet_id
        local_facet_dofs = Ferrite.facedof_indices(field_ip)[local_facet_id]
        for index in local_facet_dofs
            push!(dofs, celldofs(dh, cell_id)[index])
        end
    end

    sort!(dofs)
    unique!(dofs)
    return dofs
end

function setup_periodic_dofsets(dh::DofHandler)
    per_dofsets = Vector{Vector{Int}}(undef, 3)
    for i in 1:3
        per_dofsets[i] = get_dofs_on_facetset(dh, "periodic_$(i)a")
    end
    return per_dofsets
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

function get_periodic_dofset(disc::FEMDiscretization, field::Symbol)
    if field == :ψ
        return disc.ψ_per_dofset
    elseif field == :ρ
        return disc.ρ_per_dofset
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

    if field == :ρ
        Ferrite.apply!(H, ch)
        return H[ch.free_dofs, ch.free_dofs]
    end
    H
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
    neg_half_laplace
end

function init_refinement_matrix(disc::FEMDiscretization{T}) where T
    dh_ψ = get_dof_handler(disc, :ψ)
    dh_ρ = get_dof_handler(disc, :ρ)
    cv_ψ = get_cell_values(disc, :ψ)
    cv_ρ = get_cell_values(disc, :ρ)

    ip_ψ = Ferrite.getfieldinterpolation(dh_ψ, Ferrite.find_field(dh_ψ, :ψ))
    ip_ρ = Ferrite.getfieldinterpolation(dh_ρ, Ferrite.find_field(dh_ρ, :ρ))
    ref_coords_ρ = Ferrite.reference_coordinates(ip_ρ)

    n_basefuncs_ψ = getnbasefunctions(cv_ψ)
    ref_evals_ψ = Ferrite.reference_shape_value.([ip_ψ], ref_coords_ρ, (1:n_basefuncs_ψ)')

    ψ_inds, ρ_inds, vals = Int[], Int[], T[]
    for cell in CellIterator(dh_ρ)
        reinit!(cv_ρ, cell)

        ψ_dofs = apply_inverse_constraint_map(disc, celldofs(dh_ψ, cell.cellid), :ψ)
        ρ_dofs = apply_inverse_constraint_map(disc, celldofs(cell), :ρ)
        
        for (i_ρ, dof_ρ) in enumerate(ρ_dofs), (i_ψ, dof_ψ) in enumerate(ψ_dofs)
            push!(ψ_inds, dof_ψ)
            push!(ρ_inds, dof_ρ)
            push!(vals, ref_evals_ψ[i_ρ, i_ψ])
        end
    end
    return sparse(ρ_inds, ψ_inds, vals, get_n_free_dofs(disc, :ρ), get_n_free_dofs(disc, :ψ), (x, y) -> x)      # filters out duplicate entries. Values should be equal anyways,
                                                                                                                # but by default sparse sums them up, which we have to prevent.
end

function get_free_dof_positions(disc::FEMDiscretization{T}, field::Symbol) where T
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
        
        free_dofs = get_free_dofs(disc, field)
        free_dofs_in_cell = [(i, apply_inverse_constraint_map(disc, dof, field)) for (i, dof) in enumerate(celldofs(cell)) if dof in free_dofs]
        @views for (i, free_dof) in free_dofs_in_cell
            col = cell_dof_coords[:, i]
            dof_coords[free_dof] = Vec3{T}(col)
        end
    end
    return dof_coords
end