import NFFT

# Perform non-uniform FFTs (NFFTs) from Fourier space to real space. This is necessary
# for the construction of periodic potentials for FEM, which are not particularly nice
# in real space. It is easier to calculate them in Fourier space, then transform them
# to real space using an NFFT. In particular, we need to transform a function from a
# uniform grid in Fourier space to a non-uniform grid in real space.
#
# For some reason, the naming conventions are reversed compared to standard FFTs:
# The forward transform (nfft) goes from Fourier space to real space, and the adjoint
# transform (anfft) goes from real space to Fourier space (even though the paper below
# explicitly mentions that it is the other way around).
#
# For more information, see the NFFT.jl paper (https://arxiv.org/pdf/2208.00049).

struct NFFTGrid{T, VT <: Real}
    nfft_size::Tuple{Int, Int, Int}

    nfft
    nfft_normalization::T
    anfft_normalization::T

    architecture::AbstractArchitecture
end

function NFFTGrid(nfft_size::Tuple{Int, Int, Int}, r_vectors::AbstractVector{Vec3{VT}},
                  unit_cell_volume::T, arch::AbstractArchitecture
                 ) where {T <: Real, VT <: Real}
    r_vectors_device = to_device(arch, reduce(hcat, r_vectors))

    nfft = NFFT.plan_nfft(r_vectors_device, nfft_size)

    nfft_normalization  = 1 / sqrt(unit_cell_volume)
    anfft_normalization = sqrt(unit_cell_volume)

    NFFTGrid{T, VT}(nfft_size, nfft, nfft_normalization, anfft_normalization, arch)
end

function nfft(nfft_grid::NFFTGrid{T}, f_real::AbstractArray) where {T}
    nfft_grid.nfft * f_real * nfft_grid.nfft_normalization
end
function nfft!(f_fourier::AbstractArray, nfft_grid::NFFTGrid{T}, f_real::AbstractArray) where {T}
    mul!(f_fourier, nfft_grid.nfft, f_real)
    mul!(f_fourier, f_fourier, nfft_grid.nfft_normalization)
end
function anfft(nfft_grid::NFFTGrid{T}, f_fourier::AbstractArray) where {T}
    adjoint(nfft_grid.nfft) * f_fourier * nfft_grid.anfft_normalization
end
function anfft!(f_real::AbstractArray, nfft_grid::NFFTGrid{T}, f_fourier::AbstractArray) where {T}
    mul!(f_real, adjoint(nfft_grid.nfft), f_fourier)
    mul!(f_real, f_real, nfft_grid.anfft_normalization)
end
function rnfft(nfft_grid::NFFTGrid{T}, f_real::AbstractArray) where {T}
    real(nfft(nfft_grid, f_real))
end
function ranfft(nfft_grid::NFFTGrid{T}, f_fourier::AbstractArray) where {T}
    real(anfft(nfft_grid, f_fourier))
end