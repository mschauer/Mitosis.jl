
using Makie
using SparseArrays
using Mitosis

"""
Graph Laplacian of a `m×n` lattice.
"""
function laplacian(T, m, n)
    S = sparse(T(0.0)I, n*m, n*m)
    linear = LinearIndices((1:m, 1:n))
    for i in 1:m
        for j in 1:n
            for (i2, j2) in ((i + 1, j), (i, j + 1))
                if i2 <= m && j2 <= n
                    S[linear[i, j], linear[i2, j2]] -= 1
                    S[linear[i2, j2], linear[i, j]] -= 1

                    S[linear[i, j], linear[i, j]] += 1
                    S[linear[i2, j2], linear[i2, j2]] += 1
                end
            end
        end
    end
    S
end

m, n = 35, 35
W = [exp(-norm(([i,j]-[m÷3,n÷3]))^2/25.0) + 0.02randn() for i in 1:m, j in 1:n]

C = CartesianIndices((m, n))
L = LinearIndices((m, n))
i = CartesianIndex(m÷2, n÷2)
Λ = laplacian(Float32, m, n)
u = Gaussian{(:F,:Γ)}(zeros(m*n), (Λ + 0.1I)^4 )

for k in 1:16
    global i, u
    H = sparse(zeros(1, m*n))
    H[L[i]] = 1
    k = kernel(Gaussian; μ=AffineMap(H, [0.]), Σ=ConstantMap(Matrix(0.02^2*I(1))))
    _, p = Mitosis.backward(BF(), k, [W[i]])
    _, u = fuse(u, p)
    img(x) = image(reshape(x, m, n))
    _, i_ = findmax(mean(u) + 1.5./sqrt.(diag(u.Γ))) # proxy for inverse
    #_, i_ = findmax(mean(u) + 2.0sqrt.(diag(inv(Matrix(u.Γ)))))

    i = C[i_]
    println(i)
end

p1 = img(mean(u))
p2 = img(W)
_, i = findmax(W)
scatter!(p1, [i[1]], [i[2]])
scatter!(p2, [i[1]], [i[2]])
_, i = findmax(mean(u))
scatter!(p1, [C[i][1]], [C[i][2]], color=:red)

hbox(p1, p2)
