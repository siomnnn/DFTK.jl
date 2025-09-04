import NFFT
using Krylov

# Perform non-uniform FFTs (NFFTs) from Fourier space to real space. This is necessary
# for the construction of periodic functions (pseudopotentials and nlcc densities) for FEM,
# which are not particularly nice in real space. It is easier to calculate them in
# Fourier space, then transform them to real space using an NFFT. In particular, we need to
# transform a function from a uniform grid in Fourier space to a non-uniform grid
# in real space.
#
# For some reason, the naming conventions of NFFTs as used in NFFT.jl are reversed
# compared to FFTs: The forward transform goes from Fourier space to real space, and
# the adjoint transform goes from real space to Fourier space (even though the paper below
# explicitly mentions that it is the other way around).
#
# We adopt a different naming scheme that is also common in the literature: We call the
# "forward" transform "nfft2" (NFFT type 2). The adjoint transform would be called
# "nfft1" (NFFT type 1), but we will only need "nfft2".
#
# Note that nfft2 is NOT the inverse of nfft1 (although it is its adjoint).
#
# For more information, see the NFFT.jl paper (https://arxiv.org/pdf/2208.00049).

struct NFFTGrid{T, VT <: Real}
    nfft_size::Tuple{Int, Int, Int}

    nfft_vectors

    nfft1
    nfft2

    nfft1_normalization::T
    nfft2_normalization::T

    architecture::AbstractArchitecture
end

function NFFTGrid(nfft_size::Tuple{Int, Int, Int}, r_vectors::AbstractVector{Vec3{VT}},
                  unit_cell_volume::T, arch::AbstractArchitecture
                 ) where {T <: Real, VT <: Real}
    nfft_vectors = to_device(arch, NFFT_vectors(nfft_size))
    r_vectors_device = to_device(arch, reduce(hcat, r_vectors))

    nfft2 = NFFT.plan_nfft(r_vectors_device, nfft_size)
    nfft1 = adjoint(nfft2)

    nfft1_normalization  = sqrt(unit_cell_volume)
    nfft2_normalization = 1 / sqrt(unit_cell_volume)

    NFFTGrid{T, VT}(nfft_size, nfft_vectors, nfft1, nfft2, nfft1_normalization, nfft2_normalization, arch)
end

G_vectors(nfft_grid::NFFTGrid) = nfft_grid.nfft_vectors
function NFFT_vectors(nfft_size::Tuple{Int, Int, Int})
    start = .- cld.(nfft_size .- 1, 2)
    stop = fld.(nfft_size .- 1, 2)

    axes  = [collect(start[i]:stop[i]) for i = 1:3]
    [Vec3{Int}(i, j, k) for i in axes[1], j in axes[2], k in axes[3]]
end

function nfft1(nfft_grid::NFFTGrid{T}, f_real::AbstractArray) where {T}
    nfft_grid.nfft1 * f_real * nfft_grid.nfft1_normalization
end
function nfft1!(f_fourier::AbstractArray, nfft_grid::NFFTGrid{T}, f_real::AbstractArray) where {T}
    mul!(f_fourier, nfft_grid.nfft1, f_real)
    mul!(f_fourier, f_fourier, nfft1_grid.nfft1_normalization)
end
function nfft2(nfft_grid::NFFTGrid{T}, f_fourier::AbstractArray) where {T}
    nfft_grid.nfft2 * f_fourier * nfft_grid.nfft2_normalization
end
function nfft2!(f_real::AbstractArray, nfft_grid::NFFTGrid{T}, f_fourier::AbstractArray) where {T}
    mul!(f_real, nfft_grid.nfft2, f_fourier)
    mul!(f_real, f_real, nfft_grid.nfft2_normalization)
end
function rnfft1(nfft_grid::NFFTGrid{T}, f_real::AbstractArray) where {T}
    real(nfft1(nfft_grid, f_real))
end
function rnfft2(nfft_grid::NFFTGrid{T}, f_fourier::AbstractArray) where {T}
    real(nfft2(nfft_grid, f_fourier))
end