mutable struct AreaChangeOptimiser <: Optimisers.AbstractRule
    m
    minchange
    xs
    losses
    function AreaChangeOptimiser(m, minchange=0.001)
        new(m, minchange, [], [])
    end
end

function Optimisers.apply!(o::AreaChangeOptimiser, s, x, x̄)
    @unpack m, minchange, losses, xs = o
    push!(xs, x)

    @debug extrema(x̄)
    @assert all(m.p .== x)
    @assert length(xs) == length(losses)

    if length(xs) > 1 && losses[end] > losses[end-1]
        # w = 0.85
        # m.p .== w * xs[end-1] + (1 - w) * xs[end]
        # o.η /= 1.4
        o.minchange /= 1.2
    else
        o.minchange *= 1.2
    end

    A = prod(size(m))
    a0 = m() .> 0.5
    x0 = deepcopy(x)
    x̄0 = deepcopy(x̄)

    dA = 0
    i = 0
    c = 1
    while i == 0 || c < 1f38 && dA < minchange * A

        x̄ = c * A * x̄0
        m.p .= x0 - x̄

        m.p .= min.(1, m.p)
        m.p .= max.(0, m.p)

        a = m() .> 0.5
        dA = sum(abs, a - a0)

        c *= 1.5
        i += 1
    end

    x̄ = x0 - m.p
    m.p .= x0

    @debug c, dA, o.minchange
    println("fractional change: $(dA/A)")

    return s, x̄
end

function Optimisers.init(o::AreaChangeOptimiser, x)
    nothing
end
