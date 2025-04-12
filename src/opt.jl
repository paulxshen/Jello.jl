mutable struct AreaChangeOptimiser <: Optimisers.AbstractRule
    m
    change
    ρ
    x̄
    xs
    losses
    function AreaChangeOptimiser(m, change=0.001, ρ=0.1)
        new(m, change, ρ, 0, [], [])
    end
end

function Optimisers.apply!(o::AreaChangeOptimiser, s, x, x̄)
    @unpack m, change, losses, xs = o
    push!(xs, x)

    @debug extrema(m.p)
    @debug extrema(x̄)
    @assert all(m.p .== x)
    @assert length(xs) == length(losses)

    if length(xs) > 1 && losses[end] >= losses[end-1]
        # w = 0.85
        # m.p .== w * xs[end-1] + (1 - w) * xs[end]
        o.change /= 1.4
        o.ρ = 0.8o.ρ + 0.2
    else
        o.change *= 1.1
        o.ρ *= 0.95
    end

    o.x̄ = o.ρ * o.x̄ + (1 - o.ρ) * x̄
    A = prod(size(m))
    a0 = m() .> 0.5
    x0 = deepcopy(x)

    dA = 0
    i = 0
    c = 1
    while i == 0 || c < 1f38 && dA < change * A
        c *= 1.2

        x̄ = A * c * o.x̄
        m.p .= x0 - x̄

        m.p .= min.(1, m.p)
        m.p .= max.(0, m.p)

        a = m() .> 0.5
        dA = sum(abs, a - a0)

        i += 1
    end

    x̄ = x0 - m.p
    m.p .= x0

    @debug c, dA, o.change, o.ρ
    println("fractional change: $(dA/A)")

    return s, x̄
end

function Optimisers.init(o::AreaChangeOptimiser, x)
    nothing
end
