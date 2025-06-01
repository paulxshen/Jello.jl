function Blob(sz::Base.AbstractVecOrTuple;
    # lvoid=0, lsolid=0,
    lmin,
    symmetries=(), periodic=false,
    init=0.5, init_holes=false,
    F=Float32)

    init = F(init)
    sz = Tuple(sz)
    N = length(sz)
    symmetries = Symbol.(symmetries)
    # @assert lsolid > 0 || lvoid > 0
    # lmin = max(lsolid, lvoid)

    if !periodic
        σ = 0.5lmin
        R2 = round(Int, 2σ)
        R1 = round(Int, lmin / 2)
        R = R1

        if :diagonal ∈ symmetries
            psz = Tuple(fill(maximum(sz), N))
        else
            psz = sz
        end
        psz += 2R
        if init == 0.5
            # d = F(0.1)
            # p = F(0.5) + 2d * (init - 1 + rand(F, sz))
            p = rand(F, psz)
        elseif init == 0
            p = zeros(F, psz)
        elseif init == 1
            p = ones(F, psz)
        else
            p = pad(init, :replicate, R)
        end
        init_holes && perforate!(p, 1.5 * 2lmin, 0.8lmin / 2)
        p = 0.8p + 0.1 |> F
        p .+= (rand(F, size(p)) - F(0.5)) / 5

        n = 2R1 + 1
        conv = Conv((n, n), 1 => 1)
        conv.weight .= ball(R1, N; normalized=true) do r
            (R1 - r + 1) / R1
            # exp(-(r / σ)^2 / 2)
        end

        return ConvBlob(p, sz, symmetries, conv)
    else
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)