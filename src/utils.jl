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
function conic(r, d; normalized=false)
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
function apply_symmetries(a, symmetries, sz)
    if isempty(symmetries)
        return a
    end
    for s = symmetries
        a = cat(a, reverse(selectdim(a, s, 1:sz[s]-size(a, s)), dims=s), dims=s)
    end
    a
end

function apply_symmetries(a, symmetries)
    for d = symmetries
        d = string(d)
        if startswith(d, "diag")
            a = (a + a') / 2 # diagonal symmetry in this Ceviche challenge
        elseif startswith(d, "anti")
            a = (a + reverse(a, dims=1)') / 2
            # elseif startswith(d ,"anti")
        elseif d == "inversion"
            a += reverse(a, dims=Tuple(1:ndims(a)))
            a /= 2
        else
            d = ignore_derivatives() do
                parse(Int, d)
            end
            a += reverse(a, dims=d)
            a /= 2
        end
    end
    a
end
function step(a::AbstractArray{T}, α::Real) where {T}
    m = a .> 0.5
    α = T(α)
    a = α * tanh.(a - T(0.5)) + (m) * (1 - α) + (1 - m) * (α)
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

function imframe(a0, frame=nothing, margin=0)
    if isnothing(frame)
        a0
    else
        start = margin + 1
        roi = @ignore_derivatives range.(start, start + size(a0) - 1)
        b = Buffer(a0, size(frame)...)
        copyto!(b, frame)
        b[roi...] = a0
        copy(b)
    end
end

function loss(a::AbstractArray{T}, lsolid=0, lvoid=0,) where {T}
    m = Array(a) .> 0.5
    n = 4
    ps = T.((1:n) / n)
    A = if lsolid > 0
        ignore_derivatives() do
            sum(ps) do p
                Rsolid = max(1, round(p * lsolid / 2 - 0.01))
                ms = opening(m, se(Rsolid, ndims(a)))
                (ms .< m) * T(1 / n)
            end
        end
    else
        0
    end
    B = if lvoid > 0
        ignore_derivatives() do
            sum(ps) do p
                Rvoid = max(1, round(p * lvoid / 2 - 0.01))
                mv = closing(m, se(Rvoid, ndims(a)))
                (mv .> m) * T(1 / n)
            end
        end
    else
        0
    end
    sum(A .* a + B .* (1 - a)) / prod(size(a))
end
function smooth(a::T, α=0, sesolid=nothing, sevoid=nothing) where {T}
    m0 = Array(a) .> 0.5
    m = m0

    A, B = ignore_derivatives() do
        if !isnothing(sesolid)
            m = opening(m, sesolid)
        end
        if !isnothing(sevoid)
            m = closing(m, sevoid)
        end
        m .> m0, m .< m0
    end
    a + (1 - α) * T(A - B)
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