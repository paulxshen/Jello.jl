# Define a container to hold any optimiser specific parameters (if any):
mutable struct SurrogateOptimiser <: Optimisers.AbstractRule
    opt
    nn
    xs
    x̄s
    ls
    Zxs
    Zx̄s
    ntrain
    nmax
    ntrust
    function SurrogateOptimiser(opt=Adam())
        new(opt, nothing, [], [], [], 1, 1, 1, 100, 0)
    end
end

function Optimisers.apply!(o::SurrogateOptimiser, s, x, x̄)
    if isnothing(o.nn)
        n = length(x)
        n1 = round(n / 50)
        n1 = 2n
        n2 = 4n
        o.nn = Chain(Dense(n, n1, relu), Dense(n1, n2, relu), Dense(n2, n))
        # o.nn = Dense(n, n)
        # o.nn = identity
    end

    @unpack xs, x̄s, ls, nn, ntrain, nmax = o
    n = length(xs)
    if n < nmax
        push!(xs, x)
        push!(x̄s, x̄)
        n += 1
    end
    @assert n == length(x̄s) == length(ls)

    if n < ntrain
    elseif n == ntrain
        o.ntrust = 1
    elseif ls[end] < ls[end-1]
        o.ntrust += 1
    else
        o.ntrust = max(0, o.ntrust - 1)
    end

    if o.ntrust > 0
        println("training surrogate")
        l = Inf
        o.Zx̄s = mean(mean.(abs, x̄s))
        o.Zxs = mean(mean.(abs, xs))
        opt = Adam()
        opt_state = Flux.setup(opt, nn)
        i = 0
        while i < 200 && l > 0.1
            l, (dldm,) = Flux.withgradient(nn) do m
                mean(map(x̄s / o.Zx̄s, xs / o.Zxs) do x̄, x
                    Flux.mae(x̄, m(x))
                end)
            end
            if i % 10 == 0
                println("training loss $l")
            end
            Flux.update!(opt_state, nn, dldm)
            i += 1
        end
    end


    s, x̄ = Optimisers.apply!(o.opt, s, x, x̄)
    # println("surrogate taking $(o.ntrust) stepfuncs")
    # for i = 1:(1+o.ntrust)
    #     s, x̄ = Optimisers.apply!(o.opt, s, x, x̄)
    #     if i > 1
    #         x̄ = nn(x / o.Zxs) * o.Zx̄s
    #     end
    # end
    return s, x̄
end

function Optimisers.init(o::SurrogateOptimiser, x)
    opt_state = Optimisers.init(o.opt, x)
end

