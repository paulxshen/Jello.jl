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

    if !periodic
        σ = 0.5lmin
        R2 = round(Int, 2σ)
        R1 = round(Int, lmin / 2)
        R = R1
        psz = sz + 2R
        if isa(init, Real)
            # d = F(0.1)
            # p = F(0.5) + 2d * (init - 1 + rand(F, sz))
            p = rand(F, psz)
        else
            p = pad(init, :replicate, R)
        end
        p = 0.8p + 0.1 |> F

        n = 2R1 + 1
        conv1 = Conv((n, n), 1 => 1)
        conv1.weight .= ball(R1, N; normalized=true) do r
            (R1 - r + 1) / R1
            # exp(-(r / σ)^2 / 2)
        end

        n = 2R2 + 1
        conv2 = Conv((n, n), 1 => 1)
        conv2.weight .= ball(R2, N; normalized=true) do r
            exp(-(r / σ)^2 / 2)
        end

        return ConvBlob(p, symmetries, conv1, conv2)
    else
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)