function Blob(sz::Base.AbstractVecOrTuple;
    # lvoid=0, lsolid=0,
    lmin,
    symmetries=(), periodic=false,
    init=0.5,
    F=Float32)

    init = F(init)
    sz = Tuple(sz)
    N = length(sz)
    # @assert lsolid > 0 || lvoid > 0
    # lmin = max(lsolid, lvoid)

    margins = ceil(Int, sz / 2)
    asz = sz + margins

    psz = ceil.(Int, asz / lmin / 2) + 1
    psz = min.(psz, round.(Int, asz / 2 + 0.1))
    if isa(init, Real)
        init = rand(F, asz) .< init
    end

    p = fft(pad(2 * (init - F(0.5)), :replicate, margins))[range.(1, psz)...]
    p .*= 2
    p[1] /= 2

    mask = map(CartesianIndices(p)) do I
        norm((collect(Tuple(I)) - 1) ./ (size(p) - 1)) <= 1.001
    end
    p = stack([real(p), imag(p)])

    return FourierBlob(p, sz, asz, mask, symmetries)
end
Blob(sz...; kw...) = Blob(sz; kw...)
