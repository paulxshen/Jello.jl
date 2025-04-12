struct ConvBlob
    p::AbstractArray
    symmetries
    conv1
    conv2
end
Base.size(m::ConvBlob) = size(m.p)
foo(x) = x
function (m::ConvBlob)()
    @unpack p, symmetries, conv1, conv2 = m
    # p = _ConvBlob(p, symmetries, conv1, 0.1)
    p = _ConvBlob(p, symmetries, conv1)
    # @debug p |> extrema
    foo(p)
end

function _ConvBlob(a::AbstractArray{T}, symmetries, conv, α=0.001) where T
    @nograd (conv,)
    a = apply_symmetries(a, symmetries)

    N = ndims(a)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))

    stepfunc.(a, α)
end
function ChainRulesCore.rrule(::typeof(foo), y::AbstractArray{T}) where T
    function pb(ȳ)
        # Z = maximum(abs, ȳ)
        # if Z > 0
        #     ȳ /= Z
        # end
        r = y - T(0.5)
        NoTangent(), ((ȳ .> 0) .== (r .> 0)) .* ȳ #.* sqrt.(abs.(r))
        # NoTangent(), ȳ
    end
    return y, pb
end