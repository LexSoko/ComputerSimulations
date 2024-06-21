using Pkg
Pkg.add(["GLMakie", "Profile", "LinearAlgebra", "SparseArrays", "ImageMagick"])
using Profile
using GLMakie
using LinearAlgebra
using SparseArrays
BLAS.set_num_threads(6)
GLMakie.activate!()
# gr()
# Time-dependent Schrödinger equation in 2D

function ψ0(x, y, kx, ky, Δk)
  return Δk / sqrt(π) * exp.(-(Δk .* x) .^ 2 ./ 2 .+ im * kx .* x) * transpose(exp.(-(Δk .* y) .^ 2 ./ 2 + im * ky .* y))
end


# function buildr(ψn, Δx, Δt, V)
#   r = zeros(ComplexF64, length(ψn))
#   r[1] = ψn[1] .+ im * Δt / 2 * ((ψn[2] .- 2 .* ψn[1]) / Δx^2 .- V[1] .* ψn[1])
#   @inbounds for i in 2:length(ψn)-1
#     r[i] = ψn[i] .+ im * Δt / 2 * ((ψn[i+1] .+ ψn[i-1] .- 2 .* ψn[i]) / Δx^2 .- V[i] .* ψn[i])
#   end
#   r[end] = ψn[end] .+ im * Δt / 2 * ((-2 .* ψn[end] .+ ψn[end-1]) / Δx^2 .- V[end] .* ψn[end])
#   return r
# end

function laplacian2D(Nx, Ny, Δx, Δy, Δt, V)
  rx = im * Δt / (2 * Δx^2)
  ry = im * Δt / (2 * Δy^2)
  V_ = im * Δt / 2 * reshape(V, Nx * Ny)
  # Slow as hell
  # lp = spzeros(ComplexF64, (Nx * Ny, Nx * Ny))
  # lp[diagind(lp, 0)] .= 1 .+ 2 * rx .+ 2 * ry
  # lp[diagind(lp, 1)] .= -rx
  # lp[diagind(lp, -1)] .= -rx
  # lp[diagind(lp, Nx)] .= -ry
  # lp[diagind(lp, -Nx)] .= -ry
  rows = Int64[]
  cols = Int64[]
  vals = ComplexF64[]
  for i in 1:(Nx*Ny)
    if i > Nx
      push!(rows, i)
      push!(cols, i - Nx)
      push!(vals, -ry)
    end

    if i <= Nx * Ny - Nx
      push!(rows, i)
      push!(cols, i + Nx)
      push!(vals, -ry)
    end

    if i % Nx != 1
      push!(rows, i)
      push!(cols, i - 1)
      push!(vals, -rx)
    end
    if i % Nx != 0
      push!(rows, i)
      push!(cols, i + 1)
      push!(vals, -rx)
    end

    push!(rows, i)
    push!(cols, i)
    push!(vals, 1 + 2 * rx + 2 * ry + V_[i])
  end
  lp = sparse(rows, cols, vals, Nx * Ny, Nx * Ny)
  return lp
end

function doubleSlit2DPotential(width, gap_size, intergapdistance, V0, xstart, ycenter, xrange, yrange)
  V = zeros(Float32, length(xrange), length(yrange))
  @inbounds for i in eachindex(xrange)
    if abs(xrange[i] - xstart) < width / 2
      @inbounds for j in eachindex(yrange)
        if abs(yrange[j] - ycenter - intergapdistance / 2) > gap_size / 2 && abs(yrange[j] - ycenter + intergapdistance / 2) > gap_size / 2
          V[i, j] = V0
        end
      end
    end
  end
  return V
end

# sparse Cranck Nicolson method
function crackNicSparse()
  kx = 5π
  ky = 0
  Δk = 1
  t0 = 0
  tf = 2.0
  Δx = 0.05
  Δy = 0.05
  Δt = Δx^2 / 2
  t = collect(t0:Δt:tf)
  x = collect(-5:Δx:15)
  y = collect(-10:Δy:10)



  # b
  lap = laplacian2D(length(x), length(y), Δx, Δy, Δt, zeros(ComplexF64, length(x), length(y)))
  H = lap
  F = lu(H, check=true)
  L = F.L
  U = F.U

  ψ_s = ψ0(x, y, kx, ky, Δk)
  ψ_ = reshape(ψ_s, length(x) * length(y))
  ψ = (F.Rs.*ψ_)[F.p]
  magψ0 = sum(abs2.(ψ)) .* Δx .* Δy
  @assert (reshape(ψ_, length(x), length(y)) ≈ ψ_s)
  @assert reshape(ψ[invperm(F.p)] ./ F.Rs, (length(x), length(y))) ≈ ψ_s

  # testing inverse of vector
  # b = L \ ψ
  # z = U \ b
  # @assert isapprox(L * U * z, ψ)

  H̃_ = laplacian2D(length(x), length(y), Δx, Δy, -Δt, zeros(ComplexF64, length(x), length(y)))
  # H̃ = H̃ + spdiagm(0 => V)
  @assert isapprox(F.Rs .* ψ_, ψ[invperm(F.p)])
  @assert isapprox(ψ_, ψ[invperm(F.p)] ./ F.Rs)
  @assert isapprox((F.Rs.*(H̃_*(ψ[invperm(F.p)]./F.Rs)))[F.p], (F.Rs.*(H̃_*ψ_))[F.p])

  fig, ax1, hm = heatmap(x, y, abs2.(reshape(ψ[invperm(F.q)], length(x), length(y))), colormap=:hot)
  ax1.title = "Free particle"
  norm_timeseries = zeros(Float32, length(t))
  record(fig, "ex14_free_particle.mkv", t) do t_
    r = (F.Rs.*(H̃_*(ψ[invperm(F.q)])))[F.p]
    b = L \ r
    ψ = U \ b
    ψ_reshaped = reshape(ψ[invperm(F.q)], (length(x), length(y)))
    hm[3] = abs2.(ψ_reshaped)
  end

  fig, ax = lines(t, norm_timeseries)
  # add labels
  ax.xlabel = "Time"
  ax.ylabel = "Norm squared"
  ax.title = "Time series of the norm squared"
  # set limits y axis
  ylims!(ax, (minimum(norm_timeseries), 1e-17))

  save("ex14_norm_timeseries_free_particle.png", fig)

  # c
  V = doubleSlit2DPotential(0.5, 0.5, 1, 250, 5, 0, x, y)
  # display(V)
  lap = laplacian2D(length(x), length(y), Δx, Δy, Δt, V)
  H = lap
  F = lu(H, check=true)
  L = F.L
  U = F.U

  ψ_s = ψ0(x, y, kx, ky, Δk)
  ψ_ = reshape(ψ_s, length(x) * length(y))
  ψ = (F.Rs.*ψ_)[F.p]
  magψ0 = sum(abs2.(ψ)) .* Δx .* Δy

  H̃_ = laplacian2D(length(x), length(y), Δx, Δy, -Δt, V)
  @assert isapprox(F.Rs .* ψ_, ψ[invperm(F.p)])
  @assert isapprox(ψ_, ψ[invperm(F.p)] ./ F.Rs)
  @assert isapprox((F.Rs.*(H̃_*(ψ[invperm(F.p)]./F.Rs)))[F.p], (F.Rs.*(H̃_*ψ_))[F.p])

  fig = Figure()
  ga = fig[1, 1] = GridLayout()
  # fig, ax1, hm = heatmap(x, y, abs2.(reshape(ψ[invperm(F.q)], length(x), length(y))), colormap=:hot)
  ax1, hm = heatmap(ga[1, 1], x, y, abs2.(reshape(ψ[invperm(F.q)], length(x), length(y))), colormap=:hot)
  ax1.title = "Double slit potential"

  # ax2, lp = lines(fig[2, 1], [-Δt, 0], [1, normsqrd])
  # ax2, lp = lines(gb[1, 1], [-Δt, 0], [0, normsqrd])
  # ll = Observable([Point2f(0, normsqrd)])
  # ll = Observable(Point2f[(0, normsqrd)])
  # ax2, lp = lines(gb[1, 1], ll)

  # println(typeof(lp))
  # println(lp)
  # println(typeof(lp[1]))
  # println(typeof(lp[1][]))
  norm_timeseries = zeros(Float32, length(t))
  record(fig, "ex14_doubleSlit.mkv", collect(1:length(t))) do i
    r = (F.Rs.*(H̃_*(ψ[invperm(F.q)])))[F.p]
    b = L \ r
    ψ = U \ b

    ψ_reshaped = reshape(ψ[invperm(F.q)], (length(x), length(y)))
    # hm[3] = abs2.(ψ_reshaped) + V
    hm[3] = abs2.(ψ_reshaped)
    normsqrd = sum(abs2.(ψ ./ sqrt(magψ0))) .* Δx .* Δy .- 1.0
    norm_timeseries[i] = normsqrd


    # lp[1][] = push!(lp[1][], Point2f(t_, normsqrd))
    # ll[] = push!(ll[], Point2f(t_, normsqrd))
    # println(lp[1][])
    # push!(lp[2], normsqrd)
    # push!(lp[1], t_)
    # push!(lp[2], normsqrd)
  end

  # plot and save the time series of the norm squared

  fig, ax = lines(t, norm_timeseries)
  ax.xlabel = "Time"
  ax.ylabel = "Norm squared"
  ax.title = "Time series of the norm squared"
  ylims!(ax, (minimum(norm_timeseries), maximum(norm_timeseries)))
  save("ex14_norm_timeseries_double_slit.png", fig)

end



function main()
  crackNicSparse()
end

# https://ben.land/post/2022/03/17/complex-wavefunction-visualization/

main()
