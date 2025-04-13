
_ceil(x) = x == floor(Int, x) ? Int(x) + 1 : ceil(Int, x)
function ball(f, R, N=2; normalized=false)
    a = map(Base.product(fill(-R:R, N)...)) do r
        r = norm(collect(r))
        r > R + 0.001 ? 0 : f(r)
    end
    if normalized
        a /= sum(a)
    end
    a
end
function circle(r, d=2; normalized=false)
    n = round(Int, r - 0.01)
    a = [
        begin
            d = norm(v) - r
            if d < -0.5
                1
            elseif d > 0.5
                0
            else
                0.5 - d
            end
        end for v = Base.product(fill(-n:n, d)...)
    ] # circle
    if normalized
        a /= sum(a)
    end
    a
end
function cone(r, d; normalized=false)
    r = round(Int, r - 0.01)
    a = [1 - norm(v) / (r + 1) for v = Base.product(fill(-r:r, d)...)] # circle
    a = max.(a, 0)
    if normalized
        a /= sum(a)
    end
    a
end
function se(r, d=2)
    a = ball(x -> 1, r, d) .> 0
    # display(heatmap(a))
    centered(a)
end
function apply_symmetries(a, symmetries)
    if isempty(symmetries)
        return a
    end
    for s = symmetries
        # if :diagonal == s
        #     a += a'
        #     a /= 2
        #     # a = (a + reverse(a, dims=1)') / 2
        #     # a += reverse(a, dims=Tuple(1:ndims(a)))
        #     # a /= 2
        # end
        if Symbol(s) == :diagonal
            a += a'
            a /= 2
        else
            dims = findfirst(==(string(s)), ["x", "y", "z"])
            if !isnothing(dims)
                a += reverse(a; dims)
                a /= 2
                # a = cat(a, reverse(selectdim(a, s, 1:sz[s]-size(a, s)), dims=s), dims=s)
            end
        end
    end
    # if :inversion ∈ symmetries
    #     for dims = 1:ndims(a)
    #         a += reverse(a; dims)
    #         a /= 2
    #     end
    # end
    a
end

Base.round(x::AbstractArray) = round.(x)
function ChainRulesCore.rrule(::typeof(round), x)
    y = round(x)
    function pb(ȳ::T) where T
        # T = eltype(y)
        # r = x - 0.5 |> T
        # NoTangent(), ((ȳ .> 0) .== (r .> 0)) .* ȳ #.* sqrt.(abs.(r))
        # NoTangent(), ȳ .* (2abs.(r) + 0.01) |> T
        NoTangent(), ȳ
    end
    return y, pb
end

# foo(x) = x
# # foo(x) = Base.round.(x)
# function Zygote.rrule(::typeof(foo), x)
#     y = foo(x)
#     function pb(ȳ)
#         @debug extrema(ȳ)
#         # T = eltype(x)
#         # r = x - T(0.5)
#         # NoTangent(), ((ȳ .> 0) .== (r .> 0)) .* ȳ #.* sqrt.(abs.(r))
#         NoTangent(), ȳ
#     end
#     y, pb
# end