struct ConvBlob
    p::AbstractArray
    symmetries
    conv
end
Base.size(m::ConvBlob) = size(m.p)
function (m::ConvBlob)()
    @unpack p, symmetries, conv = m
    # p = _ConvBlob(p, symmetries, conv, 0.1)
    p = _ConvBlob(p, symmetries, conv)
    # @debug p |> extrema
    # foo(p)
end

function _ConvBlob(a::AbstractArray{T,N}, symmetries, conv) where {T,N}
    @nograd (conv,)
    a = apply_symmetries(a, symmetries)
    # @debug a |> extrema

    R = (size(conv.weight)[1] - 1) ÷ 2
    a = reshape(a, size(a)..., 1, 1)
    a = conv(a)
    a = dropdims(a, dims=(N + 1, N + 2))

    gl = pad.(diff.((a,), 1:N), :replicate, [1:N .== i for i = 1:N], 0)
    gr = pad.(diff.((a,), 1:N), :replicate, 0, [1:N .== i for i = 1:N])  # Added missing definition for gr
    al = (a,) .- gl / 2
    ar = (a,) .+ gr / 2

    b = [a, a]
    v = map(1:N) do i
        map(CartesianIndices(a)) do I
            x = b[i][I]
            l = al[i][I]
            r = ar[i][I]

            # v = map(al, ar) do al, ar
            #     map(al, ar, a) do l, r, x

            _x = x > 0.5
            _l = l > 0.5
            _r = r > 0.5  # Fixed variable name

            TOL = 1.0f-6
            ((_l == _x || abs(x - l) < TOL ? _x : (max(x, l) - T(0.5)) / abs(x - l)) +
             (_x == _r || abs(x - r) < TOL ? _x : (max(x, r) - T(0.5)) / abs(r - x))) / 2
        end
    end
    a = map(v...) do v...
        for (i, x) = enumerate(v)
            (0 < x < 1 || i == length(v)) && return x
        end
    end
    # @debug a |> extrema
    @assert minimum(a) >= 0 && maximum(a) <= 1
    a
    # _a .* .!edges + a .* edges
end
