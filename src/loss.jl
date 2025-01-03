function tent(x, a, b)
    m = (a + b) / 2
    if x < m
        (x - a) / (m - a)
    else
        1 - (x - m) / (b - m)
    end
end
function loss(Avalssolid, Avalsvoid, rvalssolid, rvalsvoid, ker, a, frame, start)
    # @nogradvars (ker, levels, frame)
    # if !isnothing(frame)
    # a = a0
    # a = if isnothing(frame)
    #     a0
    # else
    #     roi = @ignore_derivatives range.(start, start + size(a0) - 1)
    #     b = Buffer(a0, size(frame)...)
    #     copyto!(b, frame)
    #     b[roi...] = a0
    #     copy(b)
    #     # w, h = size(a0)
    #     # m = start[1] - 1
    #     # frame = @ignore_derivatives frame
    #     # l, r = @ignore_derivatives frame[1:m, m+1:m+h], frame[m+w+1:end, m+1:m+h]
    #     # a1 = vcat(l, a0, r)
    #     # l, r = @ignore_derivatives frame[:, 1:m], frame[:, m+h+1:end]
    #     # hcat(l, a1, r)
    # end
    r = @ignore_derivatives int((size(ker, 1) - 1) / 2)
    mask = @ignore_derivatives Gray.(a .> 0.5)
    Is, Iv = ignore_derivatives() do
        map((mask - erode(mask), dilate(mask) - mask)) do edges
            I = map(findall(>(0.5), edges)) do I
                I = Tuple(I)
                range.(I - r, I + r)
            end
            filter(I) do I
                checkbounds(Bool, a, I...)
            end
        end
    end
    losssolid, lossvoid = map(
        ((Is, Avalssolid, rvalssolid, false),
        (Iv, Avalsvoid, rvalsvoid, true))
    ) do (I, Avals, rvals, inv)
        isempty(I) && return 0
        a1 = if inv
            1 - a
        else
            a
        end
        mean(I) do I
            x = a1[I...] ⋅ @ignore_derivatives ker
            if x < Avals[end]
                r = getindexf(rvals, indexof(x, Avals))
                return tent(r, 0, rvals[end])
            end
            0
        end
    end
    # (; a, loss, losssolid, lossvoid, mask, edges)
end