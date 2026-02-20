using Pkg

Pkg.activate(@__DIR__) # use docs/Project.toml
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, ".."))) # local package
Pkg.instantiate()

using KeemenaLM
using Documenter
using Documenter: DocMeta

DocMeta.setdocmeta!(KeemenaLM, :DocTestSetup, :(using KeemenaLM); recursive=true)

makedocs(
    modules = [KeemenaLM],
    sitename = "KeemenaLM.jl",
    authors = "Alexander V. Mantzaris",
    format = Documenter.HTML(; canonical = "https://mantzaris.github.io/KeemenaLM.jl", edit_link = "main"),
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(
    repo = "github.com/mantzaris/KeemenaLM.jl",
    devbranch = "main",
)
