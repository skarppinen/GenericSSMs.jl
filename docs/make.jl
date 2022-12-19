using GenericSSMs, Documenter

makedocs(sitename = "GenericSSMs.jl",
         pages = ["Introduction" => "index.md", 
                  "Installation & quick start" => "quick-start.md",
                  "State-space models and Feynman-Kac representations" => "ssms.md",
                  "Defining SSMs using GenericSSMs.jl" => "interface.md",
                  "Using GenericSSMs.jl" => "use-cases.md",
                  "Examples" => "examples.md",
                  "Resampling API" => "resampling-api.md"],
        format = Documenter.HTML(prettyurls = false))

deploydocs(
    repo = "github.com/skarppinen/GenericSSMs.jl.git",
    versions = ["stable" => "v^"] # stable links to latest version
)
