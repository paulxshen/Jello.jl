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
    lvoid=0, lsolid=0, morph=true,
    symmetries=(), periodic=false, solid_frac=0.5,
    frame=nothing, start=1,
    F=Float32, T=F)
    sz = Tuple(sz)
    symmetries = Tuple(symmetries)
    N = length(sz)
    @assert lsolid > 0 || lvoid > 0
    lmin = max(lsolid, lvoid)


    rsolid = round(lsolid / 2 - 0.01) - 1
    rvoid = round(lvoid / 2 - 0.01) - 1
    sesolid = morph && rsolid > 0 ? se(rsolid, N) : nothing
    sevoid = morph && rvoid > 0 ? se(rvoid, N) : nothing

    if !periodic
        lmin /= sqrt(N)
        lmin = max(1, lmin)
        # σ = T(0.5lmin)
        # Rf = round(1.5σ - 0.01)
        Rf = round(0.5lmin - 0.01)
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
        psz = min.(asz, round(asz / lmin))

        p = rand(T, psz)
        Isolid = p .> (1 - solid_frac)
        p = Isolid .* (0.5 / solid_frac * (p - 1 + solid_frac) + 0.5) + (!(Isolid)) .* (0.5 / (1 - solid_frac) * p)
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
            # gaussian(x / σ)
            1
        end


        jump = if sesolid == sevoid == nothing
            0
        else
            minimum(sum.(filter(x -> x != nothing, [sesolid, sevoid])))
        end
        return InterpBlob(p, A, sz, asz, sesolid, sevoid, jump, frame, margin, symmetries, conv)
    else
        psz = round(sz / lmin)
        psz = min.(psz, sz)
        p = randn(T, (psz..., 2))
        solid_frac = 0.5
        p[1] = solid_frac |> T
        p[length(p)÷2+1] = 0
        p[1] *= prod(sz)

        return FourierBlob(p, sz, sesolid, sevoid, symmetries)
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

Optimisers.maywrite(x::AbstractSparseArray) = true