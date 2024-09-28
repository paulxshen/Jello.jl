_ceil(x) = x == floor(Int, x) ? Int(x) + 1 : ceil(Int, x)
function circle(r, d)
    r = round(Int, r)
    [norm(v) <= r + 0.001 for v = Base.product(fill(-r:r, d)...)] # circle
end
function conic(r, d)
    r = round(Int, r)
    a = [1 - norm(v) / (r + 1) for v = Base.product(fill(-r:r, d)...)] # circle
    a /= sum(a)
end
function se(r, d=2)
    centered(circle(round(r), d))
end
function apply(symmetries, r)
    if !isempty(symmetries)
        for d = symmetries
            d = string(d)
            if startswith(d, "diag")
                r = (r + r') / 2 # diagonal symmetry in this Ceviche challenge
            elseif startswith(d, "anti")
                r = (r + reverse(r, dims=1)') / 2
                # elseif startswith(d ,"anti")
            elseif d == "inversion"
                r += reverse(r, dims=Tuple(1:ndims(r)))
                r /= 2
            else
                d = ignore_derivatives() do
                    parse(Int, d)
                end
                r += reverse(r, dims=d)
                r /= 2
            end
        end
    end
    r
end
function step(a::AbstractArray{T}, α::Real) where {T}
    m = a .> 0
    α = T(α)
    a = α / 2 * tanh.(a) + (m) * (1 - α / 2) + (1 - m) * (α / 2)
    # a = min.(1, a)
    # a = max.(-1, a)
end

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

function smooth(a, α, lvoid=0, lsolid=0, frame=nothing, lmin=0)
    if lvoid == lsolid == 0
        return a
    end

    rvoid = round(lvoid / 2 + 0.01)
    rsolid = round(lsolid / 2 + 0.01)

    T = typeof(a)
    m0 = Array(a) .> 0.5
    m = m0

    A, B = ignore_derivatives() do
        #     for (ro, rc) in zip(ropen:-1:1, rclose:-1:1)
        #         # for (ro, rc) in zip(1:ropen, 1:rclose)
        #         # a = openclose(a, ro, rc)
        #         m = closing(m, se(rc, ndims(a)))
        #         m = opening(m, se(ro, ndims(a)))
        #     end
        if !isnothing(frame)
            o = fill(round(lmin) + 1, ndims(m))
            roi = range.(o, o + size(m) - 1)
            _m = copy(frame)
            _m[roi...] = m
            m = _m
        end

        if rsolid > 0
            m = opening(m, se(rsolid, ndims(a)))
        end
        if rvoid > 0
            m = closing(m, se(rvoid, ndims(a)))
        end
        if !isnothing(frame)
            m = m[roi...]
        end
        m .> m0, m .< m0
    end
    A, B = T.((A, B))
    a + (1 - α) * (A - B)
end

function resize(a, sz)
    if length(sz) == 1
        return imresize(a, sz, method=ImageTransformations.Lanczos4OpenCV())
    end
    imresize(a, sz)
end

