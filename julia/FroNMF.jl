# FroNMF.jl
# - Author: Takehiro Sano
# - License: GNU General Public License v3.0

using LinearAlgebra
using SparseArrays
using Random

function mur_solver(X, _W, _H; eps=1e-9, max_iter=100)
    W = deepcopy(_W)
    H = deepcopy(_H)
    E = X - W * H'
    obj_per_time = [(0.0, 0.5 * norm(E)^2)]
    start = time()

    for _ in 1:max_iter
        # Update
        W = W .* (X * H) ./ (W * (H' * H))
        W = max.(eps, W)
        H = H .* (X' * W) ./ (H * (W' * W))
        H = max.(eps, H)

        E = X - W * H'
        push!(obj_per_time, (time()-start, 0.5 * norm(E)^2))
    end

    return W, H, obj_per_time
end

function hals_solver(X, _W, _H; delta=1e-9, max_iter=100)
    W = deepcopy(_W)
    H = deepcopy(_H) 
    E = X - W * H'
    obj_per_time = [(0.0, 0.5 * norm(E)^2)]
    start = time()

    row, n_components = size(W)

    for _ in 1:max_iter
        # Update
        for k in 1:n_components
            R = E + W[:, k] .* H[:, k]'
            W[:, k] = max.(0.0, R * H[:, k] + delta * W[:, k]) / (dot(H[:, k], H[:, k]) + delta)
            w_nrm = norm(W[:, k])
            if w_nrm > 0.0
                W[:, k] /= w_nrm
            else
                W[:, k] = rand(MersenneTwister(1234), row)
                W[:, k] /= norm(W[:, k])
            end

            H[:, k] = max.(0.0, R' * W[:, k]) 

            E = R - W[:, k] .* H[:, k]'
        end

        push!(obj_per_time, (time()-start, 0.5 * norm(E)^2))
    end

    return W, H, obj_per_time
end

function _gcd_solver(X, W, H, Wnew, P, Z, G, S, D, q, delta)
    n_samples, n_components = size(W)

    mul!(P, H', H)
    mul!(Z, X, H) 
    mul!(G, W, P)
    G = G .- Z 

    for r in 1:n_components
        for i in 1:n_samples
            S[i, r] = max(0.0, W[i, r] - G[i, r] / (delta + P[r, r])) - W[i, r]
            D[i, r] = - G[i, r] * S[i, r] - 0.5 * P[r, r] * S[i, r]^2
        end
    end

    p_init = -1.0
    for i in 1:n_samples
        qi = argmax(D[i, :])
        q[i] = qi
        p_init = max(p_init, D[i, qi])
    end

    fill!(Wnew, 0.0)
    nu = 0.001

    for i in 1:n_samples
        for _ in 1:n_components^2
            qi = q[i]
            if D[i, qi] < nu * p_init
                break
            end

            s = S[i, qi]
            Wnew[i, qi] += s
            G[i, :] = G[i, :] .+ (s .* P[qi, :])

            for r in 1:n_components
                S[i, r] = max(0.0, W[i, r] - G[i, r] / (delta + P[r, r])) - W[i, r]
                D[i, r] = - G[i, r] * S[i, r] - 0.5 * P[r, r] * S[i, r]^2
            end

            q[i] = argmax(D[i, :])
        end
    end
    copyto!(W, W + Wnew)
    W = max.(0.0, W)
end

function gcd_solver(X, _W, _H; delta=1e-9, max_iter=100)
    W = deepcopy(_W)
    H = deepcopy(_H)
    E = X - W * H'
    row, col = size(X) 
    n_components = size(W)[2]
    obj_per_time = [(0.0, 0.5 * norm(E)^2)]
    start = time()

    Wnew = zeros(row, n_components)
    PWW = zeros(n_components, n_components)
    PXW = zeros(row, n_components)
    GW = zeros(row, n_components)
    SW = zeros(row, n_components)
    DW = zeros(row, n_components)
    qW = Array{Int}(undef, row)
    #
    Hnew = zeros(col, n_components)
    PHH = zeros(n_components, n_components)
    PXH = zeros(col, n_components)
    GH = zeros(col, n_components)
    SH = zeros(col, n_components)
    DH = zeros(col, n_components)
    qH = Array{Int}(undef, col)

    for _ in 1:max_iter
        # Update
        _gcd_solver(X, W, H, Wnew, PWW, PXW, GW, SW, DW, qW, delta)
        for k in 1:n_components
            w_nrm = norm(W[:, k])
            H[:, k] *= w_nrm
            if w_nrm > 0.0
                W[:, k] /= w_nrm
            else
                W[:, k] = rand(MersenneTwister(1234), row)
                W[:, k] /= norm(W[:, k])
            end
        end
        _gcd_solver(X', H, W, Hnew, PHH, PXH, GH, SH, DH, qH, 0.0)
        E = X - W * H'
        push!(obj_per_time, (time()-start, 0.5 * norm(E)^2))
    end

    return W, H, obj_per_time
end

