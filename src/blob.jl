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
    symmetries=(), periodic=false,
    init=0.5,
    frame=nothing, start=1,
    F=Float32, T=F)

    init = T(init)
    sz = Tuple(sz)
    symmetries = Tuple(symmetries)
    N = length(sz)
    @assert N == 2
    @assert lsolid > 0 || lvoid > 0
    lmin = max(lsolid, lvoid)

    rsolid = round(lsolid / 2 - 0.01) - 1
    rvoid = round(lvoid / 2 - 0.01) - 1
    sesolid = morph && rsolid > 0 ? se(rsolid, N) : 0
    sevoid = morph && rvoid > 0 ? se(rvoid, N) : 0

    if !periodic
        Rf = round(0.6lmin - 0.01)
        lgrid = lmin / 2
        lgrid = max(1, lgrid)
        # σ = T(0.5lmin)
        # Rf = round(1.5σ - 0.01)

        asz = collect(sz)
        symmetries = map(symmetries) do s
            try
                parse(Int, string(s))
            catch
                Symbol(s)
            end
        end
        asz = Tuple(asz)

        psz = min.(asz, round(asz / lgrid))
        psz = max.(1, psz)

        J = LinearIndices(psz)
        I = LinearIndices(asz)
        if asz == psz
            A = 1
        else
            ei = Base.product(fill([-1, 1], N)...) / sqrt(N) |> vec |> F
            A = map(CartesianIndices(asz)) do i
                v = T.(0.5 + (Tuple(i) - 0.5) .* psz ./ asz)
                v = max.(1, v)
                v = min.(v, psz)

                t = map(v) do x
                    f = floor(Int, x)
                    c = ceil(Int, x)
                    if f == c
                        if f == 1
                            c += 1
                        else
                            f -= 1
                        end
                    end
                    f, c
                end
                f = getindex.(t, 1)
                c = getindex.(t, 2)
                js = Base.product(zip(f, c)...)
                o = mean(js) |> F
                qs = [v - o] .⋅ ei
                V = (qs .> 0) .* qs + (1 - sum(abs, qs) / 2) / 4
                # @assert sum(V) ≈ 1

                # js = vec(Base.product(unique.(zip(floor.(Int, v), ceil.(Int, v)))...))
                # zs = norm.(collect.((v,) .- js))
                # Z = sum(zs)
                # n = length(js)
                # stack(vec([[
                #     # I[i], J[j...], prod((cospi.(j - v) + 1) / 2)
                #     I[i], J[j...], prod(1 - abs.(j - v))
                #     # I[i], J[j...], n == 1 ? 1 : (Z - z) / Z / (n - 1)
                # ] for j = js]))
                # ] for (j, z) = zip(js, zs)]))
                (
                    fill(I[i], length(js)),
                    [J[j...] for j in js] |> vec,
                    # [prod(1 - abs.(j - v)) for j in js] |> vec
                    V
                )
            end
            I = reduce(vcat, getindex.(A, 1))
            J = reduce(vcat, getindex.(A, 2))
            V = reduce(vcat, getindex.(A, 3))

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
        if isa(init, Real)

            d = T(0.1)
            p = T(0.5) + 2d * (init - 1 + rand(T, psz))
        else
            p = Array(A) \ vec(init)
            p = reshape(p, psz)
        end

        n = 2Rf + 1
        conv = Conv((n, n), 1 => 1)
        conv.weight .= ball(Rf, N; normalized=true) do x
            (Rf - x + 1) / Rf
        end

        return InterpBlob(p, A, sz, asz, sesolid, sevoid, frame, symmetries, conv)
    else
        psz = round(sz / lmin)
        psz = min.(psz, sz)
        p = randn(T, (psz..., 2))
        init = 0.5
        p[1] = init |> T
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