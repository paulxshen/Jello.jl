# - `alg`: `:interpolation` `:fourier`
"""
    Blob(sz; contrast=20, alg=:interpolation, T=Float32, rmin=nothing, lsolid=rmin, lvoid=rmin, symmetries=[])
    (m::ConvBlob)()

Functor for generating length scale controlled geometry mask, represented as array of values between 0 to 1.0. `ConvBlob` constructor makes the callable functor which can be passed as the model parameter to `Flux.jl`. Caliing it yields geometry whose length, spacing and radii are roughly on order of `lmin`. contrast controls the edge sharpness. Setting `rmin` applies additional morphological filtering which eliminates smaller features and radii

Args
- `sz`: size of mask array

Keyword Args
- `contrast`: edge sharpness
- `rmin`: minimal radii during morphological filtering, can also be `nothing` (no filtering)
- `lsolid`: same as `rmin` but only applied to solid (bright) features
- `lvoid`: ditto
- `symmetries`: symmetry dimensions
"""
function Blob(sz::Base.AbstractVecOrTuple;
    lvoid=0, lsolid=0,
    symmetries=[], periodic=false, init=nothing,
    frame=nothing, start=1,
    F=Float32, T=F)

    N = length(sz)
    @assert lsolid > 0 || lvoid > 0
    lmin = max(lsolid, lvoid)
    # lmin /= sqrt(N)
    lmin = max(1, lmin)

    if !periodic
        σ = T(0.5lmin)
        Rf = round(1.5σ - 0.01)
        if isnothing(frame)
            margin = 0
        else
            margin = 2Rf
            if isa(frame, Number)
                frame = fill(frame, sz + 2margin)
            else
                frame = frame[range.(start - margin, start + sz + margin - 1)...]
            end
        end

        asz = collect(sz)
        for s = symmetries
            asz[s] = round(asz[s] / 2 + 0.01)
        end
        asz = Tuple(asz)
        psz = min.(asz, round(2asz / lmin))

        p = if isnothing(init)
            rand(T, psz)
        elseif isa(init, Number)
            @assert init ∈ (0, 1)
            init + (-1)^(init) * 0.1rand(T, psz)
        end
        p = T.(p)

        J = LinearIndices(p)
        I = LinearIndices(asz)
        if asz == psz
            A = 1
        else
            A = map(CartesianIndices(Tuple(asz))) do i
                r = T.(0.5 + (Tuple(i) - 0.5) .* psz ./ asz)
                r = max.(1, r)
                r = min.(r, psz)

                js = collect(Base.product(Set.(zip(floor(r), ceil(r)))...))
                # zs = norm.(collect.((r,) .- js))
                # Z = sum(zs)
                # n = length(js)
                stack(vec([[
                    # I[i], J[j...], prod((cospi.(j - r) + 1) / 2)
                    I[i], J[j...], prod(1 - abs.(j - r))
                    # I[i], J[j...], n == 1 ? 1 : (Z - z) / Z / (n - 1)
                ] for j = js]))
                # ] for (j, z) = zip(js, zs)]))
            end
            A = reduce(hcat, vec(A))
            A = sparse(eachrow(A)...)
        end
        p = vec(p)

        n = 2Rf + 1
        conv = Conv((n, n), 1 => 1)
        conv.weight .= ball(Rf, N; normalized=true) do x
            gaussian(x / σ)
        end
        return InterpBlob(p, A, sz, asz, lmin, lvoid, lsolid, frame, margin, symmetries, conv)
    else
        psz = round(sz / lmin)
        if isnothing(init)
            pre = randn(T, psz)
            pim = randn(T, psz)
        else
            pre = zeros(T, psz)
            pim = zeros(T, psz)
            if init == 1
                pre[1] = sqrt(length(pre))
            end
        end

        return FourierBlob(pre, pim, sz, lsolid, lvoid, symmetries)
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)
# if verbose
#     @info """
# Blob configs

# Geometry generation 
# - algorithm: Fourier basis
# - Fourier k-space size (# of Fourier basis per dimension): $nbasis
# $com
# """


# Rk = 2
# ker = circle(Rk, N)
# rvalssolid = T.(collect(range(0, Rsolid, 8)))
# rvalsvoid = T.(collect(range(0, Rvoid, 8)))
# Avalssolid = ica.(Rk, rvalssolid, rvalssolid - 0.5)
# Avalsvoid = ica.(Rk, rvalsvoid, rvalsvoid - 0.5)
# @show A = sum(ker), π * Rk^2