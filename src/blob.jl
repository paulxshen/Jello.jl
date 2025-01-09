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
    lvoid=0, lsolid=0, morph=false,
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
    sesolid = morph && rsolid > 0 ? se(rsolid, N) : 0
    sevoid = morph && rvoid > 0 ? se(rvoid, N) : 0

    if !periodic
        # Rf = round(0.8lmin - 0.01)
        # if isnothing(frame)
        #     margin = 0
        # else
        #     margin = 2Rf
        #     if isa(frame, Number)
        #         frame = fill(frame, sz + 2margin)
        #     else
        #         frame = frame[range.(start - margin, start + sz + margin - 1)...]
        #     end
        # end

        # psz = collect(sz)
        # for s = symmetries
        #     psz[s] = round(psz[s] / 2 + 0.01)
        # end
        # psz = Tuple(psz)
        # psz = max.(1, psz)

        # p = rand(T, psz)
        # Isolid = p .> (1 - solid_frac)
        # p = Isolid .* (0.5 / solid_frac * (p - 1 + solid_frac) + 0.5) + (!(Isolid)) .* (0.5 / (1 - solid_frac) * p)
        # p = T.(p)

        # n = 2Rf + 1
        # conv = Conv((n, n), 1 => 1)
        # conv.weight .= ball(Rf, N; normalized=true) do x
        #     (Rf - x + 1) / Rf
        # end

        # return ConvBlob(p, sz, sesolid, sevoid, frame, margin, symmetries, conv)

        # elseif false

        Rf = round(0.8lmin - 0.01)
        lgrid = lmin / 3
        lgrid = max(1, lgrid)
        # σ = T(0.5lmin)
        # Rf = round(1.5σ - 0.01)
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

        psz = min.(asz, round(asz / lgrid))
        psz = max.(1, psz)

        p = rand(T, psz)
        p = p .> (1 - solid_frac)
        # p += 0.1randn(T, size(p))
        p = T.(p)

        J = LinearIndices(p)
        I = LinearIndices(asz)
        if asz == psz
            A = 1
        else
            A = map(CartesianIndices(asz)) do i
                v = T.(0.5 + (Tuple(i) - 0.5) .* psz ./ asz)
                v = max.(1, v)
                v = min.(v, psz)

                js = collect(Base.product(unique.(zip(floor(v), ceil(v)))...))
                # zs = norm.(collect.((v,) .- js))
                # Z = sum(zs)
                # n = length(js)
                stack(vec([[
                    # I[i], J[j...], prod((cospi.(j - v) + 1) / 2)
                    I[i], J[j...], prod(1 - abs.(j - v))
                    # I[i], J[j...], n == 1 ? 1 : (Z - z) / Z / (n - 1)
                ] for j = js]))
                # ] for (j, z) = zip(js, zs)]))
            end
            A = reduce(hcat, vec(A))

            I, J, V = eachrow(A)
            m = prod(asz)
            n = prod(psz)

            I = round.(Int, I)
            J = round.(Int, J)
            I = min.(I, m)
            I = max.(I, 1)
            J = min.(J, n)
            J = max.(J, 1)
            A = sparse(I, J, V, m, n)
        end
        p = vec(p)

        n = 2Rf + 1
        conv = Conv((n, n), 1 => 1)
        conv.weight .= ball(Rf, N; normalized=true) do x
            (Rf - x + 1) / Rf
        end

        return InterpBlob(p, A, sz, asz, sesolid, sevoid, frame, margin, symmetries, conv)
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
holesize(m::InterpBlob) = (2^length(m.symmetries)) * maximum(sum, (m.sesolid, m.sevoid))