function Blob(sz::Base.AbstractVecOrTuple;
    lmin,
    symmetries=(), periodic=false,
    init=0.5,
    init_pattern_level=nothing, init_pattern_spacing=nothing, init_pattern_diameter=nothing,
    F=Float32)

    init = F(init)
    sz = Tuple(sz)
    N = length(sz)
    symmetries = Symbol.(symmetries)
    if !isnothing(init_pattern_level)
        if isnothing(init_pattern_spacing)
            init_pattern_spacing = 2lmin
        end
        if isnothing(init_pattern_diameter)
            init_pattern_diameter = init_pattern_spacing / 2
        end
    end
    if !periodic
        if lmin == 1
            psz = sz
        else
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
        end
        if init == 0.5
            # d = F(0.1)
            # p = F(0.5) + 2d * (init - 1 + rand(F, sz))
            p = rand(F, psz)
        elseif init == 0
            p = zeros(F, psz)
        elseif init == 1
            p = ones(F, psz)
        else
            p = init
        end
        b = zeros(Bool, psz)
        b[(:).(3, size(b) - 2)...] .= 1
        p = ifelse.(b, p, 1 - p)
        p = pad(p, :replicate, R)
        if Symbol(init_pattern_level) == :invert
            b = zeros(Bool, size(p))
            perforate!(b, 1, init_pattern_spacing, init_pattern_diameter, R)
            p = ifelse.(b, 1 .- p, p)

        elseif !isnothing(init_pattern_level)
            perforate!(p, init_pattern_level, init_pattern_spacing, init_pattern_diameter, R)
        end
        p = 0.8p + 0.1 |> F
        p .+= (rand(F, size(p)) - F(0.5)) / 5

        n = 2R1 + 1
        W = ball(R1, N; normalized=true) do r
            (R1 - r + 1) / R1
            # exp(-(r / σ)^2 / 2)
        end |> F

        return ConvBlob(p, sz, symmetries, W)
    else
    end
end
Blob(sz...; kw...) = Blob(sz; kw...)