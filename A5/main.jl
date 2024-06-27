import Pkg;
#Pkg.update("GLMakie")
#Pkg.update("LinearAlgebra")
#Pkg.update("Distributions")
#Pkg.update("ProgressBars")

using CSV, Tables
using LsqFit
using GLMakie
using LinearAlgebra
using Distributions
using ProgressBars
using Colors
using ColorSchemes
# Constants

kb = 1.3806e-23
ϵ = 125.7 * kb
σ = 0.3345 * 1e-9
mass = 39.98 * 1.660 * 1e-27

#got fucntion from the web because there is no implementaiton in julia i suppose
function myhist(data, min, max, nbins)
    N = length(data)             # How many elements in the input vector 'data' ?
    delta = (max-min)/nbins      # Bin size is inferred here from the maximal, minimal, and bin number
    out = zeros(nbins)           # Let's initialize the output data structures for the bin count
    bin = zeros(nbins)           # and for the bin centres...
       
    start = min                  # Left edge
    for k in tqdm(1:nbins)
      stop   = start + delta   # Right edge
      out[k] = count(x -> x >= start && x < stop, data) # Count how many elements are between left and right 
      bin[k] = start + delta/2. # Centre of the bin
      start  = stop            # New left edge
     end
     return out, bin
    end
  
  
#vel_dist(vel,p) = @. 4*pi*(vel^2)*((1/(2*pi*p[1])))^(3/2) *exp(-0.5*(vel^(2))/p[1])
function vel_dist(vel, p)
    vel = abs.(vel) # Ensure vel is non-negative
    σ = p[1]
    4 * π * (vel .^ 2) .* ((1 / (2 * π * σ)) ^ (3/2)) .* exp.(-0.5 * (vel .^ 2) / σ)
end   
function height_dist(height,p)
    hs = p[1]
    p0 = p[2]
    p0.* exp.(-height./hs)
end
# Lenard-Jones potential function
function LenardJones(r, mass, ϵ, g)
    mag = sqrt.(sum(r.^2, dims=3))[:,:,1]
    a = (48) .* (mag.^(-14) - 2^(-1) * mag.^(-8))
    N = size(r,2) 
    G = zeros(Float64,N,size(r,3))
    for i in 1:N
        G[i,:] = [0 0 g]
    end
    
    return sum(r .* a, dims=1)[1, :, :] .-G
end
function LenardJonespot(positions,r, mass, ϵ,σ, g)
    mag = sqrt.(sum(r.^2, dims=3))[:,:,1]
    a = -(4 ) .* (mag.^(-12) - mag.^(-6))
    N = size(r,2) 
    G = zeros(Float64,N,size(r,3))
    for i in 1:N
        G[i,:] = [0 0 g]
    end
    E_g = positions.*G
    return sum(a, dims=1)[1, :, :] ,  E_g
end
# Function to calculate Delta
function calculateDelta(xn::Matrix{Float64})
    x_temp = xn
    Delta = zeros(Float64, size(xn, 1) - 1, size(xn, 1), size(xn, 2))
    for i in 1:(size(xn, 1) - 1)
        x_temp = circshift(x_temp, 1)
        Delta[i, :, :] = xn - x_temp
    end
    return Delta
end
function checkBoundary(position, velocity, boundary)
    N = size(position, 1)
    
    for i in 1:N
        change_vel = (position[i, :] .< boundary[1, :]) .| (position[i, :] .> boundary[2, :])
        indices = findall(change_vel)
        velocity[i,indices] = -velocity[i,indices]  
    end
    return velocity
end 
function set_molecule_pos(N::Int64, boundary; random = false, grid = [0 0 0], offset = [0 0 0]) 
    positions = zeros(Float64,N,size(boundary,2))
    velocities = zeros(Float64,N,size(boundary,2))
    if random
        for i in 1:N
            positions[i,1] = rand(boundary[1,1]:boundary[2,1])
            positions[i,2] = rand(boundary[1,2]:boundary[2,2])
            positions[i,3] = rand(boundary[1,3]:boundary[2,3]) 
        end
    else
        if grid[1] == 0
            return positions , velocities
        else
            n = 1
            z = boundary[1,3] + 0.1
            while n < N
                for i in 1:grid[1]
                    if n > N
                        break
                    end
                    for j in 1:grid[2]
                        if n > N
                            break
                        end
                        positions[n,:] = [float(i)*1.12, float(j)*1.12, z*1.12]
                    
                        n = n + 1
                        
                    end
                end
                z= z+1.0
            end
        end
    end
    N = size(positions,1)
    offsets =  repeat(offset', N, 1)
    positions = positions .+ offsets
    return positions , velocities
end 

# Leapfrog integration function
function Leapfrog(A::Function, nsteps::Int64, dt::Float64, x0::Matrix{Float64}, v0::Matrix{Float64},boundary::Matrix{Float64};nskip = 1)
    all_positions = zeros(Float64, nsteps, size(x0, 1), size(x0, 2))
    all_velocities = zeros(Float64, nsteps, size(x0, 1), size(x0, 2))
    all_accelerations = zeros(Float64, nsteps, size(x0, 1), size(x0, 2))
    all_positions[1, :, :] = x0
    all_velocities[1, :, :] = v0
    Delta_R = calculateDelta(all_positions[1, :, :])
    all_accelerations[1, :, :] = A(Delta_R)
    for n in tqdm(2:nsteps)
        all_positions[n, :, :] = all_positions[n-1, :, :] .+ dt .* all_velocities[n-1, :, :] .+ (0.5 * dt^(2)) .* all_accelerations[n-1, :, :] 
        Delta_R = calculateDelta(all_positions[n, :, :])
        all_accelerations[n, :, :] = A(Delta_R)
        all_velocities[n, :, :] = all_velocities[n-1, :, :] .+ (0.5 * dt) * (all_accelerations[n-1, :, :] .+ all_accelerations[n, :, :])
        all_velocities[n, :, :] = checkBoundary(all_positions[n, :, :],all_velocities[n,:,:],boundary)
        
    end
    return all_positions, all_velocities, all_accelerations
end


function interact_plot(filename,pos,vel,dt)
    fps = 1 / dt
    vel_mag = 0
    viridis = ColorSchemes.viridis
    with_theme(theme_black()) do
        time = Observable([200.0])
        energy= Observable([200.0])
        points = Observable(Point3f[])
        colors = Observable(RGBf[])
        record_flag = Observable(false)
        fig = Figure(size=(1800, 1200))
        ax = LScene(fig[1, 1], height=1000, width=1000)
        #ax2 = LScene(fig[1, 2], height=300, width=300)
        fig[2, 3] = buttongrid = GridLayout(tellwidth=false)
        scatter!(ax, Point3f(boundary[1,1],boundary[1,2],boundary[1,3]+1), markersize=1)
        scatter!(ax, Point3f(boundary[2,1],boundary[2,2],boundary[2,3]), markersize=1)
        
        buttonlabels = ["Start/Stop", "Record/Stop"]
        buttons = buttongrid[1, 1:2] = [Button(fig, label=l, labelcolor=:black) for l in buttonlabels]

        new_points = Point3f[]
        new_colors = RGBf[]

        N = size(pos, 2)
        max_vel = maximum(sqrt.(sum(vel.^2, dims=3)))

        for i in 1:N
            push!(new_points, Point3f(pos[1, i, 1], pos[1, i, 2], pos[1, i, 3]))
            vel_mag = sqrt(sum(vel[1, i, :] .^ 2))
            
            color = get(ColorSchemes.viridis, vel_mag / (max_vel) + vel_mag * 0.3)
            push!(new_colors, color)
        end
        println(time,"\n", energy)
        points[] = new_points
        colors[] = new_colors

        #lines!(ax2,energy)
        scatter!(ax, points,color = colors, markersize=15)

        axis = ax.scene[OldAxis]
        axis[:aspectmode] = "data"
        axis[:names, :axisnames] = ("x", "y", "z")
        tstyle = axis[:names]
        tstyle[:fontsize] = 10
        tstyle[:textcolor] = (:red, :green, :blue)
        tstyle[:font] = "helvetica"
        tstyle[:gap] = 5
        axis[:ticks][:textcolor] = :white
        axis[:ticks][:fontsize] = 5

        function update_points(frame)
            new_points = Point3f[]
            new_colors = RGBf[]
            #new_Energies = Energy[1:frame] # Corrected here
            #new_time = Time[1:frame] # Corrected here
            N = size(pos, 2)
            for i in 1:N
                push!(new_points, Point3f(pos[frame, i, 1], pos[frame, i, 2], pos[frame, i, 3]))
                vel_mag = sqrt(sum(vel[frame, i, :] .^ 2))

                color = get(ColorSchemes.viridis, vel_mag / max_vel + vel_mag * 0.3)
                push!(new_colors, color)
            end
            #time[] = new_time
            #energy[]= new_Energies 
            points[] = new_points
            colors[] = new_colors
        end

        
        slider = Slider(fig, range=1:size(pos, 1), width=1000)

        on(slider.value) do frame
            update_points(frame)
        end

        function record_animation(filename)
            record(fig, filename, 1:size(pos, 1); framerate=fps) do i
                println(100*i/size(pos,1))
                update_points(i)
                #sleep(1/fps)
            end
        end

        on(buttons[2].clicks) do n
            if record_flag[]
                record_flag[] = false
            else
                record_flag[] = true
                @async record_animation(filename)
            end
        end
        #fig[1,2] = ax2
        #fig[2, 1] = slider_anim_speed
        fig[1, 1] = ax
        fig[2, 1] = slider

        fig
        display(fig)
    end
end
function plot_energy(totEnergy,Ekin, Epot_L, Epot_G, time; filename = "default")
    fig = Figure(size=(1600,900))
    ax = Axis(fig[1, 1],
        yautolimitmargin = (0.1, 0.1),
        xautolimitmargin = (0.1, 0.1),
        title = "Energies",
        xlabel = "timesteps",
        ylabel = "Energy")

    lines!(ax,
        time,
        totEnergy,
        color =:blue,
        label = L"E_{tot}",
        linewidth = 2)
    axislegend(ax;labelsize=50)

    ax = Axis(fig[1, 2],
        yautolimitmargin = (0.1, 0.1),
        xautolimitmargin = (0.1, 0.1),
        title = "Energies",
        xlabel = "timesteps",
        ylabel = "Energy")
    
    lines!(ax,
        time,
        Ekin,
        color = :red,
        label = L"E_{kin}",
        linewidth = 2)
    axislegend(ax;labelsize=50)
    ax = Axis(fig[2, 1],
        yautolimitmargin = (0.1, 0.1),
        xautolimitmargin = (0.1, 0.1),
        title = "Energies",
        xlabel = "timesteps",
        ylabel = "Energy")
    
    lines!(ax,
        time,
        Epot_L,
        color = :green,
        label = L"E_{pot_L}",
        linewidth = 2)
    axislegend(ax;labelsize=50)
    ax = Axis(fig[2, 2],
        yautolimitmargin = (0.1, 0.1),
        xautolimitmargin = (0.1, 0.1),
        title = "Energies",
        xlabel = "timesteps",
        ylabel = "Energy")
    lines!(ax,
        time,
        Epot_G,
        label = L"E_{pot_G}",
        linewidth = 2)
            
    
    axislegend(ax;labelsize=50)
    display(fig)
    fig
    if filename != "default"
        save("$filename.png", fig)
    end
end
function plot_n_fit_vel_dist(all_vel, ntherm,σ ; fit = false ,filename = "default")
    figv = Figure(size = (1600,900))
    N = size(all_vel,2)
    ax2 = Axis(figv[1, 1], 
        yautolimitmargin = (0.1, 0.1), 
        xautolimitmargin = (0.1, 0.1),
        title = "Velocity Distribution of $N Particles",
        xlabel = L"\frac{|\vec{v}|}{\sigma} (s^{-1})",
        ylabel = L"p(|\vec{v}|)"
        )
   
    vel_mag = sqrt.(sum(all_vel .^ 2, dims=3))[:,:,1]
    vel_mag_mean = round(mean(vel_mag), digits=3)
    vel_mag_std = round(std(vel_mag), digits = 3)
    last_velocities = vel_mag[ntherm:end,:]
    num_timesteps = size(last_velocities,1)
    num_molecules = size(last_velocities,2)
    t = zeros(num_timesteps*num_molecules)
    t1 = 1
    for i in 1:num_timesteps
        for j in 1:num_molecules
            t[t1] = last_velocities[i,j]
            t1 = t1 + 1
        end
    end
    v = range(minimum(t),maximum(t),length = 1000)
    out, bin = myhist(t,minimum(t),maximum(t),100)
    
    
    normalized = out / (num_timesteps*num_molecules*(bin[2]-bin[1]))
    p0 = Float64[1.22342342]
    println(normalized)
    fit = curve_fit(vel_dist, bin, normalized, p0)
    
    
    fit_values = vel_dist(v, fit.param)
    
    param = round(fit.param[1], digits = 3)
    Temp = param*mass/kb
    println("fitparam:", param)
    println("Temp:", Temp)
    
    param_error = round(estimate_errors(fit)[1], digits = 3)
    hist!(ax2, 
        t,
        normalization = :pdf, 
        bins = 100,
        color = :values,
        label = "Histogram")

    lines!(ax2, v, fit_values, color = :red, label = "Fit param kT/m = $param +- $param_error \n T = $Temp K \n <v^2> = $vel_mag_mean +- $vel_mag_std")
    axislegend()
    if filename != "default"
        save("$filename.png", figv)
    end
    figv
    display(figv)
    
end

function calculateEnergies(all_pos,all_vel,mass,ϵ , σ,g, nskip,n_therm)
    pos = all_pos[n_therm:end,:,:]
    vel = all_vel[n_therm:end,:,:]
    N = size(pos, 1)
    pos = pos 
    vel = vel
    Ekin = sum(sum(vel .^ 2, dims=3)[:,:,1] ,dims=2)[:,1] ./2
    E_pot_L = zeros(Float64, size(pos,1))
    E_pot_g = zeros(Float64, size(pos,1))
    for i in tqdm(1:N)
        e_L, e_g = LenardJonespot(pos[i,:,:],calculateDelta(pos[i,:,:]),mass,ϵ,σ,g)
        E_pot_L[i] = sum(sum(e_L ,dims = 2)[:,1,:])
        E_pot_g[i] = sum(sum(e_g ,dims = 2)[:,1,:])
        
    end
    
    totalE= Ekin .+ E_pot_L .+ E_pot_g
    return totalE, Ekin, E_pot_L ,E_pot_g
    
end

function height_dist_plot_n_fit(all_positions,ntherm,g;filename = "default")
    pos = all_positions[ntherm:end,:,:]
    z_pos = pos[:,:,3]
    N = size(all_positions,2)
    z_pos = reshape(z_pos, size(z_pos,1)*size(z_pos,2))
    figh = Figure(size = (1600,900))
    
    ax2 = Axis(figh[1, 1], 
        yautolimitmargin = (0.1, 0.1), 
        xautolimitmargin = (0.1, 0.1),
        title = "Velocity Distribution of $N Particles",
        xlabel = L"\frac{|\vec{v}|}{\sigma} (s^{-1})",
        ylabel = L"p(|\vec{v}|)"
        )
   


    
    h = range(minimum(z_pos),maximum(z_pos),length = 1000)
    out, bin = myhist(z_pos,minimum(z_pos),maximum(z_pos),100)
    normalized = out / (length(z_pos)*(bin[2]-bin[1]))
    p0 = [10.0,10.0]
   
    fit = curve_fit(height_dist, bin, normalized, p0)
    
    fit_values = height_dist(h, fit.param)
    
    param = round(fit.param[1], digits = 3)
    kbTM = param * g
    println("fitparam:", param)
    param_error = round(estimate_errors(fit)[1], digits = 3)
    hist!(ax2, 
    z_pos,
    normalization = :pdf, 
    bins = 100,
    color = :values,
    label = "Histogram")
    lines!(ax2, h, fit_values, color = :red, label = "Fit param hs = $param +- $param_error \n kbT/m = $kbTM")
    axislegend()
    if filename != "default"
        save("$filename.png", figh)
    end
    figh
    display(figh)
    return ax2
    

end

#main simulation velosity distribution and height distribution 
#animation can be controlled via the slider
sim1 =true
if sim1 == true
    m = 1
    g = 50
    lenard(r) = LenardJones(r, m, 1, g)
    boundary = [0.0 0.0 0.0; 5.0*1.12 5.0*1.12 20]
    dt = 0.001
    N_max = 10000
    n_therm = Int(N_max*0.2)
    start_pos , start_vel = set_molecule_pos(150,boundary,random=false , grid = [4 4 5], offset = [0.0,0.0,2.0])
   
    pos, vel, acc = Leapfrog(lenard, N_max, dt, start_pos , start_vel,boundary,nskip= 30)
    
    with_theme(theme_black()) do 
        height_dist_plot_n_fit(pos,n_therm,g,filename = "heightdist_N150_50g_100K")
    end
    
    with_theme(theme_black()) do 
        plot_n_fit_vel_dist(vel,n_therm,σ,filename = "veltdist_N150_50g_100K")  
    end   

    
    interact_plot("animation5.mp4",pos,vel,dt) 
    end

#computational effort
sim2 = true
if sim2 == true
    m = 1
    g = 0.1
    lenard(r) = LenardJones(r, m, 1, g)
    N = 1:100
    println(N)
    boundary = [0.0 0.0 0.0; 10.0*1.12 10.0*1.12 15]
    dt = 0.02
    N_max = 10000
    times = zeros(Float64, N[end])
    #pos, vel, acc = Leapfrog(lenard, N_max, dt, [0.0 0.0 0.0],[0.0 0.0 0.0],boundary)
    for n in tqdm(N)
        start_pos2 , start_vel2 = set_molecule_pos(n,boundary,random=false , grid = [3 3 5], offset = [0.0,0.0,4.0])
       
        t = time()
        pos2, vel2, acc2 = @time Leapfrog(lenard, N_max, dt, start_pos2,start_vel2,boundary)
        dt1 = time() - t 
        times[n] = dt1
    end
    function compeff(x,p)
        p[1].*x.*log.(10,x)
    end

    fit2 = curve_fit(compeff, N,times, [2.0])

    fitvals = compeff(N,fit2.param)
    param = round(fit2.param[1],digits = 2)

    fig = Figure(size=(1200,800))
    ax = Axis(fig[1, 1],
    yautolimitmargin = (0.1, 0.1),
    xautolimitmargin = (0.1, 0.1),
    title = "Computational effort for 10000 timesteps",
    xlabel = "N",
    ylabel = "time elapsed")

    lines!(ax, N, times,label = "total runtime of leapfrog")
    lines!(ax, N, fitvals,label = " $param*x*log(x)")

    axislegend()    
    save("computational_effort_fit.png", fig)
    CSV.write("computational_effort_fit.csv",  Tables.table(times), writeheader=false)
    display(fig)
end