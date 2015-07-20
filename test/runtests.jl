using PanelFactorModels

tests = ["fitvariable", 
		 "fitmodel"
		 ]

println("Running tests:")

for t in tests
    tfile = string(t, ".jl")
    println(" * $(tfile) ...")
    include(tfile)
end