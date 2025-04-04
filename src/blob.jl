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
        Rf = round(2σ)
        symmetries = string.(symmetries)
        if isa(init, Real)
            # d = F(0.1)
            # p = F(0.5) + 2d * (init - 1 + rand(F, sz))
            p = rand(F, sz)
        else
            p = init
        end

        n = 2Rf + 1
        conv = Conv((n, n), 1 => 1)
        conv.weight .= ball(Rf, N; normalized=true) do r
            # (Rf - x + 1) / Rf
            exp(-(r / σ)^2 / 2)
        end

        return ConvBlob(p, symmetries, conv,)
    else
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)