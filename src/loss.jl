function tent(x, a, b)
    m = (a + b) / 2
    if x < m
        (x - a) / (m - a)
    else
        1 - (x - m) / (b - m)
    end
end
function loss(levels, ker, a0, frame, o)
    # @ignore_derivatives_vars (ker, levels, frame)
    # if !isnothing(frame)
    a = if isnothing(frame)
        a0
    else
        # roi = @ignore_derivatives range.(o, o + size(a) - 1)
        # roi = @ignore_derivatives [i:j for (i, j) = zip(o, o + size(a) - 1)]
        # b = Buffer(a, size(frame)...)
        # copyto!(b, frame)
        # b[roi...] = a
        # a = copy(b)
        w, h = size(a0)
        m = o[1] - 1
        frame = @ignore_derivatives frame
        l, r = @ignore_derivatives frame[1:m, m+1:m+h], frame[m+w+1:end, m+1:m+h]
        a1 = vcat(l, a0, r)
        l, r = @ignore_derivatives frame[:, 1:m], frame[:, m+h+1:end]
        hcat(l, a1, r)
    end
    r = @ignore_derivatives int((size(ker, 1) - 1) / 2)
    edges = ignore_derivatives() do
        mask = Gray.(a .> 0.5)
        Is = map(findall(>(0.5), mask - erode(mask))) do I
            I = Tuple(I)
            range.(I - r, I + r)
        end
        filter(Is) do I
            checkbounds(Bool, a, I...)
        end
    end
    isempty(edges) && return 0
    mean(edges) do I
        x = a[I...] ⋅ @ignore_derivatives ker
        # @show I, v
        if x <= 0
            0
        elseif x <= levels[2]
            tent(x, levels[1], levels[2])
        elseif x <= levels[3]
            0
        elseif x <= levels[4]
            tent(x, levels[3], levels[4])
        else
            0
        end
    end
end