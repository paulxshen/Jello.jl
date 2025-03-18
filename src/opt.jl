# Define a container to hold any optimiser specific parameters (if any):
mutable struct AreaChangeOptimiser <: Optimisers.AbstractRule
    opt
    m
    η
    A
    minchange
    maxchange
    xs
    ls
    function AreaChangeOptimiser(m;
        opt=Momentum(1, 0.0),
        #  opt=Adam(1, (0.8, 0.9)),
        minchange=0.001,
        maxchange=0.05)
        η = 1
        A = sum(prod(size(m)))
        new(opt, m, η, A, minchange, maxchange, [], [])
    end
end

function Optimisers.apply!(o::AreaChangeOptimiser, s, x, x̄)
    @unpack m, A, opt, maxchange, minchange, ls, xs = o
    push!(xs, x)
    @assert all(m.p .== x)
    @assert length(xs) == length(ls)
    if length(xs) > 1 && ls[end] > ls[end-1]
        w = 0.7
        m.p .== w * xs[end-1] + (1 - w) * xs[end]
    end
    a0 = m() .> 0.5

    T = eltype(x)
    dA = 0
    x0 = deepcopy(x)
    s, x̄ = Optimisers.apply!(opt, s, x, x̄)
    x̄0 = deepcopy(x̄)

    maxdA = max(1, maxchange * A)
    mindA = max(1, minchange * A)

    i = 0
    overshot = undershot = false
    while (0 < o.η < 1.0f16) && (dA > maxdA || dA < mindA)
        if i > 0
            c = 1.1f0
            if dA > maxdA
                o.η /= c
                overshot = true
            else
                o.η *= c
                undershot = true
            end
        end
        undershot && overshot && break

        x̄ = T(o.η * x̄0)
        m.p .= x0 - x̄
        m.p .= max.(PMIN, m.p)
        m.p .= min.(PMAX, m.p)
        a = m() .> 0.5
        dA = sum(abs, a - a0)
        i += 1
    end
    x̄ = x0 - m.p
    m.p .= x0

    @debug o.η, dA
    println("fractional change in design: $(dA/A)")

    return s, x̄
end

function Optimisers.init(o::AreaChangeOptimiser, x)
    opt_state = Optimisers.init(o.opt, x)
end

function update_loss!(o::AreaChangeOptimiser, l)
    push!(o.ls, l)
    if length(o.ls) > 1
        if o.ls[end] > o.ls[end-1]
            o.η /= 1.1
        else
            o.η *= 1.1
        end
    end
end
