function Blob(sz::Base.AbstractVecOrTuple;
    # lvoid=0, lsolid=0,
    lmin,
    symmetries=(), periodic=false,
    init=0.5,
    F=Float32)

    init = F(init)
    sz = Tuple(sz)
    N = length(sz)
    @assert N == 2
    # @assert lsolid > 0 || lvoid > 0
    # lmin = max(lsolid, lvoid)

    if !periodic
        Rf = round(0.6lmin - 0.01)
        lgrid = lmin / 2
        lgrid = max(1, lgrid)

        asz = collect(sz)
        symmetries = string.(symmetries)
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
                v = F.(0.5 + (Tuple(i) - 0.5) .* psz ./ asz)
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

            d = F(0.1)
            p = F(0.5) + 2d * (init - 1 + rand(F, psz))
        else
            p = Float64.(A) \ Float64.(vec(init)) .|> F
            p = reshape(p, psz)
        end

        n = 2Rf + 1
        conv = Conv((n, n), 1 => 1)
        conv.weight .= ball(Rf, N; normalized=true) do x
            (Rf - x + 1) / Rf
        end

        return InterpBlob(p, A, sz, asz, symmetries, conv)
    else
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)
Optimisers.maywrite(x::AbstractSparseArray) = true