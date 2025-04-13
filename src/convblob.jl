struct ConvBlob
    p::AbstractArray
    symmetries
    conv1
    conv2
end
Base.size(m::ConvBlob) = size(m.p)
function (m::ConvBlob)()
    @unpack p, symmetries, conv1, conv2 = m
    # p = _ConvBlob(p, symmetries, conv1, 0.1)
    p = _ConvBlob(p, symmetries, conv1)
    # @debug p |> extrema
    # foo(p)
end

function _ConvBlob(a::AbstractArray{T}, symmetries, conv) where T
    @nograd (conv,)
    a = apply_symmetries(a, symmetries)

    N = ndims(a)
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))

    round(a)
end
