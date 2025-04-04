
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
        dims = findfirst(==(string(s)), ["x", "y", "z"])
        if !isnothing(dims)
            a += reverse(a; dims)
            a /= 2
            # a = cat(a, reverse(selectdim(a, s, 1:sz[s]-size(a, s)), dims=s), dims=s)
        end
    end
    if :diagonal ∈ symmetries
        a += a'
        a /= 2

        # a += reverse(a, dims=1)'
        # a /= 2
    end
    # if :inversion ∈ symmetries
    #     for dims = 1:ndims(a)
    #         a += reverse(a; dims)
    #         a /= 2
    #     end
    # end
    a
end

# function apply_symmetries(a, symmetries)
#     for d = symmetries
#         d = string(d)
#         if startswith(d, "diag")
#             a = (a + a') / 2 # diagonal symmetry in this Ceviche challenge
#         elseif startswith(d, "anti")
#             a = (a + reverse(a, dims=1)') / 2
#             # elseif startswith(d ,"anti")
#         elseif d == "inversion"
#             a += reverse(a, dims=Tuple(1:ndims(a)))
#             a /= 2
#         else
#             d = ignore_derivatives() do
#                 parse(Int, d)
#             end
#             a += reverse(a, dims=d)
#             a /= 2
#         end
#     end
#     a
# end
function stepfunc(a::T, α) where {T}
    α = T(α)
    # a = (m) .* (1 - 2α + 2α * a) + (!m) .* (2 * α * a)
    α * tanh(a - T(0.5)) + (a > 0.5 ? 1 - α : α)
    # α * tanh(a) + (a > 0 ? 1 - α : α)
end
# stepfunc([1])

# function stepfunc(a::AbstractArray{T}) where {T}
#     m = a .> 0.5
#     α = T(0.001)
#     a = (m) .* (1 - 2α + 2α * a) + (!m) .* (2 * α * a)
#     m .* (a - a + 1)
#     # NNlib.sigmoid.(a - T(0.5))
# end
# function ChainRulesCore.rrule(::typeof(stepfunc), a)
#     y = stepfunc(a)
#     function pb(ȳ)
#         println("stepfunc")
#         # NoTangent(), ȳ .* 1 ./ (1 + 5 * abs.(2a - 1))
#         NoTangent(), ȳ
#     end
#     return y, pb
# end
# function open(a, r)
#     A = ignore_derivatives() do
#         T = typeof(a)
#         m0 = Array(a) .> 0.5
#         m = opening(m0, se(r, ndims(a)))
#         T(m0 .== m)
#     end
#     a .* A
# end
# function close(a, r)
#     A, B = ignore_derivatives() do
#         T = typeof(a)
#         m0 = Array(a) .> 0.5
#         m = closing(m0, se(r, ndims(a)))
#         T(m0 .== m), T(m .> m0)
#     end
#     a .* A + B
# end


function smooth(a::T, sesolid=0, sevoid=0) where {T}
    sesolid == sevoid == 0 && return a
    m0 = Array(a) .> 0.5
    m = m0

    m = ignore_derivatives() do
        if sesolid != 0
            m = opening(m, sesolid)
        end
        if sevoid != 0
            m = closing(m, sevoid)
        end
        m
    end
    a .* (m .== m0) + (m .> m0)
end

function resize(a, sz)
    if length(sz) == 1
        return imresize(a, sz, method=ImageTransformations.Lanczos4OpenCV())
    end
    imresize(a, sz)
end
function ica(R, r, d)
    (R == 0 || r == 0) && return 0
    (R >= r + d) && return π * r^2
    (r >= R + d) && return π * R^2
    r^2 * acos((d^2 + r^2 - R^2) / (2d * r)) + R^2 * acos((d^2 + R^2 - r^2) / (2d * R)) - sqrt((R + r + d) * (R + r - d) * (R - r + d) * (r + d - R)) / 2

end